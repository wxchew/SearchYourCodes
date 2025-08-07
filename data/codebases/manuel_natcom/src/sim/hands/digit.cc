// Cytosim was created by Francois Nedelec. Copyright 2007-2017 EMBL.

#include "digit.h"
#include "digit_prop.h"
#include "exceptions.h"
#include "iowrapper.h"
#include "messages.h"
#include "glossary.h"
#include "lattice.h"
#include "simul.h"
extern Random RNG;


#ifdef MULTI_LATTICE
Digit::Digit(DigitProp const* p, HandMonitor* h)
: Hand(p,h), mSite(0), mLattice(0), prop(p)
{
    set_lat_id(RNG.pint(4)+1);
}

#else
Digit::Digit(DigitProp const* p, HandMonitor* h)
: Hand(p,h), mSite(0), mLattice(0), prop(p)
{

}
#endif

/**
 This extract the Lattice pointer of the fiber, if it is used.
 
 @code
 if ( prop->use_lattice ),
 return Fiber::lattice();
 else
 return 0
 @endcode
 */
FiberLattice * Digit::getLattice(const Fiber * fib) const
{
    if ( fib  &&  prop->use_lattice )
    {
        FiberLattice * lat = fib->lattice();
        
        if ( lat == 0 )
            throw InvalidParameter("digit:use_lattice is true, but the Fiber has no lattice");
        
        if ( lat  &&  lat->unit() != prop->step_size  )
        {
            /*
             This is not necessarily a bug, but we would need to write additional code,
             to be able to handle a step_size that would be a multiple of fiber:lattice_unit.
             This would allow to effectively model independent lattices associated
             with independent proto-filaments
             */
            throw InvalidParameter("digit:step_size must be equal to fiber:lattice_unit");
        }
        return lat;
    }
    
    return 0;
}


/**
 This will decrease the lattice value at the current site,
 relocate to site `npos`, and increase the lattice value at the new site
 
 The FiberBinder is also updated.
 */
void Digit::resite(const int npos)
{
    if ( mLattice )
    {
        assert_true(mLattice->valid(npos));
#ifdef MULTI_LATTICE
        mLattice->dec(mSite,lat_id);
        mLattice->inc(npos,lat_id);
#else 
    #ifdef MANU_LATTICE
        mLattice->dec(mSite,prop->lat_val);
        mLattice->inc(npos,prop->lat_val);
    #else
        mLattice->dec(mSite);
        mLattice->inc(npos);
    #endif
#endif
    }
//    checkAbscissa();
    mSite = npos;
    FiberBinder::relocate(prop->step_size*(npos+0.5));
    //manu: I think this fails in stepP because of this +0.5 thing. I dont know why this is like this.
    assert_true(FiberBinder::betweenMP());
}


void Digit::moveTo(real a)
{
    if ( mLattice )
        resite(mLattice->index(a));
    else
        resite(floor(a/prop->step_size));
}


void Digit::moveToEndM()
{
    if ( mLattice )
        resite(mLattice->index(fiber()->abscissaM()));
    else
        resite(floor(fiber()->abscissaM()/prop->step_size));
}


void Digit::moveToEndP()
{
    if ( mLattice )
        resite(mLattice->index(fiber()->abscissaP()));
    else
        resite(floor(fiber()->abscissaP()/prop->step_size));
}


void Digit::relocate(Fiber * fib)
{
    if ( mLattice )
    {
#ifdef MULTI_LATTICE
        mLattice->dec(mSite, lat_id);
#else
    #ifdef MANU_LATTICE
        mLattice->dec(mSite, prop->lat_val);
    #else
        mLattice->dec(mSite);
    #endif
#endif
    }
    
    mLattice = getLattice(fib);
    if ( mLattice )
    {
#ifdef MULTI_LATTICE
        mLattice->inc(mSite, lat_id);
#else
    #ifdef MANU_LATTICE
        mLattice->inc(mSite,prop->lat_val);
    #else
        mLattice->inc(mSite);
    #endif
#endif
    }
    FiberBinder::relocate(fib);
}


void Digit::relocate(Fiber * fib, real a)
{
    if ( mLattice )
    {
#ifdef MULTI_LATTICE
        mLattice->dec(mSite, lat_id);
#else
    #ifdef MANU_LATTICE
        mLattice->dec(mSite, prop->lat_val);
    #else
        mLattice->dec(mSite);
    #endif
#endif
    }
    mLattice = getLattice(fib);
    if ( mLattice )
    {
        mSite = mLattice->index(a);
#ifdef MULTI_LATTICE
        mLattice->inc(mSite, lat_id);
#else
    #ifdef MANU_LATTICE
        mLattice->inc(mSite,prop->lat_val);
    #else
        mLattice->inc(mSite);
    #endif
#endif

    }
    else
        mSite = (int)floor(a/prop->step_size);
    FiberBinder::relocate(fib, prop->step_size*(mSite+0.5));
}
#ifdef MULTI_LATTICE
void print_lattice(Fiber * f)
{
    for (int i =0; i<4; i++) {
        std::cout << (f->multi_lattice[i] ? f->multi_lattice[i]->identity() : 0)<< ", ";
    }
}
#endif

int Digit::attachmentAllowed(FiberBinder & fb) const
{
    int return_value = 1;
    if ( !Hand::attachmentAllowed(fb) )
        return false;
    
    FiberLattice * lat = getLattice(fb.fiber());
    // calculate site of binding in 'mSite'
    if ( lat )
    {
        // closest 'integer' position:
        int npos = lat->index(fb.abscissa());
        
        // check occupancy
#ifdef MULTI_LATTICE
        Fiber * fb_new_attachment;
        Fiber * fb_already_attached;

        Digit * attached_hand = nullptr;
        if (otherHand())
        {
            if (otherHand()->prop->lat_val)
            {
                attached_hand = static_cast<Digit * >(otherHand());
            }
            else
                std::cerr << "You should not be using a couple made of one digit and one non-digit in MULTI_LATTICE mode" << std::endl;
        }

        // if the motor is AF/FA, it should have taken a random value from 1 to 4
#ifdef TRAP_SINGLES
        // It was a single, the lat_id must have been set to the right value already in FiberGrid::tryToAttach calling PartnerLattice
        if (!otherHand())
        {
            if (lat_id)
                return lat->vacant(npos, lat_id);
            else
                return false;
        }
        else if(!otherHand()->attached())
        {
            assert_true(lat_id);
            return lat->vacant(npos, lat_id);
        }
        
#else
        // If it is a single or a FF couple, asign a random value of lattice when binding to the fiber
        if (!ha.otherHand()||!ha.otherHand()->attached())
            return lat->vacant(npos, lat_id);
#endif

        else
        {
            fb_new_attachment = fb.fiber();
            fb_already_attached = attached_hand->fiber();

            
            unsigned int id_of_new_in_attached = fb_already_attached->get_lattice_val(fb_new_attachment);

            // SUMMARY OF THE MULTI_LATTICE CHECKS
            // 1. Check if the new fiber has interacted with the attached fiber before
            //      a) If they have, through the lattice in which the otherHand() is attached, then they can bind
            //      b) If they have, but through a different one, then may redistribute.
            // 2. Given that they have not interacted before, but the lattice of the otherHand() has been used for
            //    bridging the otherHand()->fiber with other fiber rather than this one.
            //      a) Check that there is an available lattice
            //      b) If the lattice has been used, you may redistribute the otherHand() to an available lattice in the
            //      otherHand()->fiber if it available.
            // 3. Given that they have not interacted before, and that the lattice of the otherHand() was not used for bridging,
            //    check whether there is an empty lattice in fb.fiber() to bind.
            
            
            // id_of_new_in_attached is true if these two fibers have interacted before,
            // and its value is the multi_lattice val in the attached fiber that corresponds to the digits that should bind to the new fiber, in "reality" this corresponds to the "side" of the attached fiber the new fiber is. If it is false, it means that these fibers have not interacted.
            
            
            // The lattice id of the attached hand
            unsigned int attached_hand_id = attached_hand->get_fiberlat_id();
            // attached_hand->get_fiberlat_id() returns the value of the lattice of the attached hand, in reality this would be the "side" of the attached fiber that the attached hand occupies.
            //  If this value is equal to id_of_new_in_attached, it means that the hand points in the same direction as the fiber, and therefore they can bind.
            
            if (id_of_new_in_attached)
            {
             
                // If this is true the hand is on the "wrong side"
                if(id_of_new_in_attached!=attached_hand_id)
                {
                    if (!attached_hand->prop->redistribute_on_binding)
                        return false;
                    // We return 2 to try to move to the right lattice. 
                    FiberLattice * attached_lat = fb_already_attached->lattice();
                    if (attached_lat && attached_lat->vacant(attached_hand->lattice_site(), id_of_new_in_attached+attached_hand->prop->lat_val_add))
                        return_value = 2;
                    else
                        return false;
                }
                // Now we know that these two fibers could be linked by this couple. We check whether the lattice site is available:
                unsigned int id_of_attached_in_new = fb_new_attachment->get_lattice_val(fb_already_attached);
                if (!lat->vacant(npos, id_of_attached_in_new+prop->lat_val_add))
                    return false;
            }
            // These two fibers didn't interact before. We verify that there are available lattices in both of them for the bridging
            else if (!fb_new_attachment->available_lattice() || !fb_already_attached->available_lattice())
            {
                MSG.warning("Warning: fiber with multi lattice encountered more than four neighbours\n");
                return false;
            }
            
            // In this case, the two fibers have not interacted before, but the lattice of the attached hand is already comitted for another fiber, we may set it to try to attach by moving to an uncomitted lattice in this fiber, if there is an available lattice in the new one (This has just been checked)
            else if (fb_already_attached->multi_lattice[attached_hand_id-1])
            {
                // The old else if:           else if (fb_already_attached->multi_lattice[attached_hand_id-1]&& fb_already_attached->multi_lattice[attached_hand_id-1] != fb_new_attachment)
                if (!attached_hand->prop->redistribute_on_binding)
                    return false;
                // The lattice where the attached hand should move
                int alt_lattice_id = fb_already_attached->available_lattice();
                assert_true(alt_lattice_id);
                // Check whether the lattice site is empty
                FiberLattice * attached_lat = fb_already_attached->lattice();
                if (!attached_lat->vacant(npos, alt_lattice_id+attached_hand->prop->lat_val_add))
                    return false;
                return_value = 2;
            }
            
//            else
//                    std::cout << "New in attached: " << id_of_new_in_attached <<std::endl;

        }

#else
    #ifdef MANU_LATTICE
        // Check whether the value at bit depth prop->lat_val is occupied
        if (!lat->vacant(npos,prop->lat_val)) {
    #else
        if (!lat->vacant(npos)) {
    #endif
            return false;
        }
#endif
        // further check for entanglement (if it is a couple and the other hand is already attached)
        if ( otherHand() && prop->lat_val && otherHand()->attached())
        {

            // otherhand is the one that is attached (old), from the old fiber we call the function
            // we pass the position in space of the old hand, the position of the new hand, and the
            // position of the new fiber
            return !(otherHand()->fiber()->causesEntanglement(otherHand()->pos(), fb.pos(), fb.fiber())) * return_value;
        }
    }
    return return_value;
}


void Digit::attach(FiberBinder const& fb)
{
    assert_true( fb.attached() );
    assert_true( unattached() );
    
    mLattice = getLattice(fb.fiber());

    if ( mLattice )
    {
        mSite = mLattice->index(fb.abscissa());
        // manu error
        //       assert_true( mLattice->vacant(mSite) );
#ifdef MULTI_LATTICE
        // If the other hand is not attached or does not exist, everything needed to be checked was checked already
        if (!otherHand() || !otherHand()->attached())
        {
            mLattice->inc(mSite, lat_id);
        }
        else
        {
            Fiber * fb_new_attachment = fb.fiber();
            Fiber * fb_already_attached = otherHand()->fiber();
            Hand  * attached_hand = otherHand();
            set_lat_id(fb_new_attachment->get_lattice_val(fb_already_attached));
            
            // If the lat_id is set already, then it means that there was already couples bridging these two fibers, therefore vacancy was tested in attachment_allowed
            if (lat_id)
                mLattice->inc(mSite, lat_id);
            else
            {
                // If this is the first couple bridging this particular set of fibers
                unsigned int ind = attached_hand->get_fiberlat_id();
                if (ind)
                {
                    // Pick up a lattice value in the new fiber where the couple will be attached.
                    set_lat_id(fb_new_attachment->available_lattice());

                    // This should always return an available lattice, because we check in attachmentAllowed
                    assert_true(lat_id);
                    
                    // This shuld have also been checked in attachmentAllowed
                    assert_true(mLattice->vacant(mSite, lat_id));
                    

                    fb_new_attachment->multi_lattice[get_fiberlat_id()-1] = fb_already_attached;
                    fb_already_attached->multi_lattice[ind-1] = fb_new_attachment;
                    mLattice->inc(mSite, lat_id);

                }
                else
                {
                    MSG.warning("An error occured, a motor had ind 0 even if attached");
                }
            }
        }
        
#else
    #ifdef MANU_LATTICE
        mLattice->inc(mSite,prop->lat_val);
    #else
        mLattice->inc(mSite);
    #endif
#endif

        //std::clog << "++Lattice("<<mSite<<") = "<<(int)mLattice->value(mSite)<<std::endl;
    }
    else
    {
        // use closest 'integer' position:
        mSite = (int)floor(fb.abscissa()/prop->step_size);
    }
    
    FiberBinder::locate(fb.fiber(), prop->step_size*(mSite+0.5));
    
    // reset event counters:
    nextDetach = RNG.exponential();
    haMonitor->afterAttachment(this);
#ifdef DIGITS_DEFINED_LENGTH
    max_abscissa = fiber()->abscissaP()/prop->step_size;
#endif

}


//------------------------------------------------------------------------------

void Digit::detach()
{
    if ( mLattice )
    {
#ifdef MULTI_LATTICE
        mLattice->dec(mSite, lat_id);
#else
    #ifdef MANU_LATTICE
        mLattice->dec(mSite, prop->lat_val);
    #else
        assert_true( mLattice->value(mSite) == 1 );
        mLattice->dec(mSite);
    #endif
#endif
    }
    
    Hand::detach();
}

//------------------------------------------------------------------------------

/**
 This will attempt to relocate to site `npos`,
 without checking the intermediate sites.
 
 @retval 0 : the move was done
 @retval 1 : the move aborted, because the specified destination is occupied
 
 */
int Digit::jumpTo(const int npos)
{
    assert_true( npos * prop->step_size     <= fiber()->abscissaP() );
    assert_true( (npos+1) * prop->step_size >= fiber()->abscissaM() );
    assert_true( npos != mSite );
#ifdef MULTI_LATTICE
    if ( mLattice == 0  ||  mLattice->vacant(npos,lat_id))
#else
    #ifdef MANU_LATTICE
    if ( mLattice == 0  ||  mLattice->vacant(npos,prop->lat_val))
    #else
    if ( mLattice == 0  ||  mLattice->vacant(npos) )
    #endif
#endif
    {
        resite(npos);
        return 0;
    }
    return 1;
}


int Digit::jumpToEnd(FiberEnd end)
{
    if ( mLattice )
    {
        int s = mLattice->index(fiber()->abscissaEnd(end));
        
        if ( end == PLUS_END )
        {
            while ( s * prop->step_size > fiber()->abscissaP() )
                --s;
        }
        else
        {
            while ( (s+1) * prop->step_size < fiber()->abscissaM() )
                ++s;
        }
        // Added by Manu
        // sometimes it would happen that there is a disassembly, and if there is hold_end, then
        // it might want to move to the plus end, which is already occupied by this hand, therefore it
        // should just stay where it is.
        if (mSite == s)
            return 0;
        else
            return jumpTo(s);
    }
    if ( end == PLUS_END )
        moveToEndP();
    else
        moveToEndM();
    return 0;
}

//------------------------------------------------------------------------------

/**
 Try to jump `n` sites in the PLUS_END direction,
 without checking the intermediate positions.
 
 @retval 0 : the move was done
 @retval 1 : the move aborted, because the destination is occupied
 @retval 2 : the move aborted, because the destination is outside the Fiber
 */
int Digit::jumpP(const int s)
{
    assert_true( s > 0 );
    assert_true( attached() );
    
    int npos = mSite + s;
    
    /*
     If lattice is used, we could have a special value to indicate
     ranges outside of the fibers.
     This would be easier to check than the abscissaM() and abscissaP()
     */
    if ( npos * prop->step_size > fiber()->abscissaP() )
        return 2;
    
    return jumpTo(npos);
}

/**
 Try to jump to `n` sites in the MINUS_END direction,
 without checking the intermediate positions.
 
 @retval 0 : the move was done
 @retval 1 : the move aborted, because the destination is occupied
 @retval 2 : the move aborted, because the destination is outside the Fiber
 */
int Digit::jumpM(const int s)
{
    assert_true( s > 0 );
    assert_true( attached() );
    
    int npos = mSite - s;
    
    /*
     If lattice is used, we could have a special value to indicate
     ranges outside of the fibers.
     This would be easier to check than the abscissaM() and abscissaP()
     */
    if ( (npos+1) * prop->step_size < fiber()->abscissaM() )
        return 2;
    
    return jumpTo(npos);
}

//------------------------------------------------------------------------------

/**
 Try to move to the adjacent site in the PLUS_END direction.
 
 @retval 0 : the move was done
 @retval 1 : the move aborted, because the destination is occupied
 @retval 2 : the move aborted, because the destination is outside the Fiber
 */
int Digit::stepP()
{
    assert_true( attached() );
    
    int npos = mSite + 1;
    
    /*
     If lattice is used, we could have a special value to indicate
     ranges outside of the fibers.
     This would be easier to check than the abscissaM() and abscissaP()
     */
    // Changed by manu
//    if ( npos * prop->step_size >= fiber()->abscissaP() )
    if ( npos >= int(fiber()->abscissaP() / prop->step_size) )
        return 2;
    
    return jumpTo(npos);
}

/**
 Try to move to the adjacent site in the MINUS_END direction.
 
 @retval 0 : the move was done
 @retval 1 : the move aborted, because the destination is occupied
 @retval 2 : the move aborted, because the destination is outside the Fiber
 */
int Digit::stepM()
{
    assert_true( attached() );
    
    int npos = mSite - 1;
    
    /*
     If lattice is used, we could have a special value to indicate
     ranges outside of the fibers.
     This would be easier to check than the abscissaM() and abscissaP()
     */
    if ( (npos+1) * prop->step_size <= fiber()->abscissaM() )
        return 2;
    
    return jumpTo(npos);
}

//------------------------------------------------------------------------------

/**
 Try to move `n` sites in the PLUS_END direction,
 stopping if any intermediate position is already occupied.
 
 @retval 0 : the specified number of steps was done
 @retval 1 : the move stopped at an occupied intermediate site
 @retval 2 : the end of the Fiber was encountered
 
 For 1 and 2, the Digit is relocated to the site just before the obstacle.
 */
int Digit::crawlP(const int n)
{
    assert_true( n > 0 );
    assert_true( attached() );
    
    if ( mLattice == 0 )
        return jumpTo(mSite+n);
    
    int res = 0, s = 0, npos = mSite;
    
    while ( s < n )
    {
        ++npos;
        
#ifdef MULTI_LATTICE
        if ( ! mLattice->vacant(npos,lat_id) )
#else
    #ifdef MANU_LATTICE
        if ( ! mLattice->vacant(npos,prop->lat_val) )
    #else
        if ( ! mLattice->vacant(npos) )
    #endif
#endif
        {
            res = 1;
            break;
        }
        /*
         We could define a special value of Lattice to indicate
         ranges outside of the fibers.
         */
        if ( npos * prop->step_size > fiber()->abscissaP() )
        {
            res = 2;
            break;
        }
        
        ++s;
    }
    
    if ( s )
    {
#ifdef MULTI_LATTICE
        assert_true( mLattice->vacant(mSite+s,lat_id) );
#else
#ifdef MANU_LATTICE
        assert_true( mLattice->vacant(mSite+s,prop->lat_val) );
#else
        assert_true( mLattice->vacant(mSite+s) );
#endif
#endif
        resite(mSite+s);
    }
    return res;
}


/**
 Try to move `n` sites in the MINUS_END direction,
 stopping if any intermediate position is already occupied.
 
 @retval 0 : the specified number of steps was done
 @retval 1 : the move stopped at an occupied intermediate site
 @retval 2 : the end of the Fiber was encountered
 
 For 1 and 2, the Digit is relocated to the site just before the obstacle.
 */
int Digit::crawlM(const int n)
{
    assert_true( n > 0 );
    assert_true( attached() );
    
    if ( mLattice == 0 )
        return jumpTo(mSite+n);
    
    int res = 0, s = 0, npos = mSite;
    
    while ( s < n )
    {
        --npos;
        
#ifdef MULTI_LATTICE
        if ( ! mLattice->vacant(npos,lat_id) )
#else
    #ifdef MANU_LATTICE
        if ( ! mLattice->vacant(npos,prop->lat_val) )
    #else
        if ( ! mLattice->vacant(npos) )
    #endif
#endif

        {
            res = 1;
            break;
        }
        /*
         We could define a special value of Lattice to indicate
         ranges outside of the fibers.
         */
        if ( (npos+1) * prop->step_size < fiber()->abscissaM() )
        {
            res = 2;
            break;
        }
        
        ++s;
    }
    
    if ( s )
    {
#ifdef MULTI_LATTICE
        assert_true( mLattice->vacant(mSite-s,lat_id) );
#else
#ifdef MANU_LATTICE
        assert_true( mLattice->vacant(mSite-s,prop->lat_val) );
#else
        assert_true( mLattice->vacant(mSite-s) );
#endif
#endif
        resite(mSite-s);
    }
    return res;
}

//------------------------------------------------------------------------------


/**
 The Digit normally does not move by itself
 */
void Digit::handleDisassemblyM()
{
    assert_true( attached() );
    
    
    if ( prop->hold_shrinking_end )
    {
        if ( jumpToEnd(MINUS_END) )
            detach();
    }
    else
        detach();
}


/**
 The Digit normally does not move by itself
 */
void Digit::handleDisassemblyP()
{
    assert_true( attached() );
    
    if ( prop->hold_shrinking_end )
    {
        if ( jumpToEnd(PLUS_END) )
            detach();
    }
    else
        detach();
}


//------------------------------------------------------------------------------
/**
 Detachment
 */
void Digit::stepUnloaded()
{
    assert_true( attached() );
    
    testDetachment();
}



/**
 (see @ref Stochastic)
 */
void Digit::stepLoaded(Vector const& force)
{
    assert_true( attached() );
    
    testKramersDetachment(force.norm());
}



//------------------------------------------------------------------------------
/**
 Digit::write() uses Lattice::TAG, instead of the Fiber::TAG,
 to indicate that the location is specified as a integer (mSite),
 rather than an float (abscissa).
 */
void Digit::write(Outputter& out) const
{
    out.writeSoftSpace();
    
    const Fiber * fib = fiber();
    if ( fib )
    {
        fib->writeReference(out, Fiber::TAG_LATTICE);
        out.writeFloat(fbAbs);
        out.writeInt32(mSite);
    }
    else {
        Object::writeNullReference(out);
    }
}



void Digit::read(Inputter& in, Simul& sim)
{
    Tag tag = 0;
    Object * w = sim.readReference(in, tag);
    
    if ( w )
    {
        Fiber * newfib = static_cast<Fiber*>(w);
        //std::clog << "FiberBinder::read() " << (char)tag << std::endl;
        
        if ( tag == Fiber::TAG )
        {
            PRINT_ONCE("INFO: Converting Hand to Digit\n");
            fbAbs  = in.readFloat();
            
            //set mSite to closest integral position
            if ( mLattice )
                mSite = mLattice->index(fbAbs);
            else
                mSite = (int)floor(fbAbs/prop->step_size);
        }
        else if ( tag == Fiber::TAG_LATTICE )
        {
            fbAbs  = in.readFloat();
            mSite  = in.readInt32();
        }
        else
        {
            ///\todo: we could allow binder to refer to any Mecable
            throw InvalidIO("Digit should be bound to a Fiber!");
        }
        
        // link the FiberBinder as in attach():
        if ( newfib != fbFiber )
        {
            if ( fbFiber )
                fbFiber->removeBinder(this);
            fbFiber = newfib;
            fbFiber->addBinder(this);
        }
        updateBinder();
        checkAbscissa();
    }
    else
    {
        if ( fbFiber )
            FiberBinder::delocate();
    }
    
    resetTimers();
}

#ifdef MULTI_LATTICE
void    Digit::stepUnattached(const FiberGrid& fg, Vector const & pos)
{
    Hand::stepUnattached(fg,pos);
}
#endif

    
int Digit::checkPos(int npos)
{
#ifdef MULTI_LATTICE
    if ( mLattice == 0  ||  mLattice->vacant(npos,lat_id))
    {
#else
#ifdef MANU_LATTICE
    if ( mLattice == 0  ||  mLattice->vacant(npos,prop->lat_val))
    {
#else
    if ( mLattice == 0  ||  mLattice->vacant(npos) )
    {
#endif
#endif
        return 0;}
    return 1;
}
        
int Digit::checkP()
{
    int npos = mSite + 1;
    
    // Changed by manu
//    if ( npos * prop->step_size >= fiber()->abscissaP() )
#ifdef DIGITS_DEFINED_LENGTH
    if ( npos >= max_abscissa )
    {
#else
    if ( npos >= int(fiber()->abscissaP() / prop->step_size) )
    {
#endif
        return 2;
    }
    return checkPos(npos);
}

int Digit::checkM()
{

    int npos = mSite - 1;
    
#ifdef DIGITS_DEFINED_LENGTH
    if ( npos <= 1 )
    {
#else
    if ( (npos+1) * prop->step_size <= fiber()->abscissaM() )
    {
#endif
        return 2;}
    
    return checkPos(npos);
}
#ifdef MANU_LATTICE
bool Digit::find_lattice_neighbour(unsigned int const v) const
{
    return !(mLattice->vacant(mSite,v));
}
    
#endif

#ifdef MULTI_LATTICE
#ifdef MANU_LATTICE
    unsigned int Digit::get_fiberlat_id() const{return lat_id - prop->lat_val_add;};
    void    Digit::set_lat_id(unsigned int const val){
        // If the value passed as val is zero, set the lat_id to zero.
        if (val)
            lat_id=val + prop->lat_val_add;
        else
            lat_id = 0;
    };
    
#else
    unsigned int Digit::get_fiberlat_id(){return lat_id;};
    void    Digit::set_lat_id(unsigned int const val){lat_id=val;};
#endif
#endif

#ifdef MULTI_LATTICE
bool Digit::attachmentSecondTry(FiberBinder & fb)
{
    // The availability of this lattice site has been already checked in Digit::AttachmentAllowed
    Fiber * fb_already_attached = fiber();
    Fiber * fb_new_attachment = fb.fiber();
    unsigned int id_of_new_in_attached = fb_already_attached->get_lattice_val(fb_new_attachment);
    if (!id_of_new_in_attached)
        id_of_new_in_attached = fb_already_attached->available_lattice();
    mLattice->dec(mSite, lat_id);
    set_lat_id(id_of_new_in_attached);
    mLattice->inc(mSite, lat_id);
    return true;
}
int Digit::lattice_val_add() const {return prop->lat_val_add;};
#endif


void Digit::stepUnattachedTrappedAA()
{
    assert_false(attached());
    // Twice because it could bind to either of the fibers
    nextAttach-=2*prop->binding_rate_dt;
    
    if (nextAttach<=0)
    {
        // The trapping couple
        HandMonitor * coup = trappedHaMon();
        
        // The hands of the trapping couple
        Hand * h1 = 0, * h2 = 0;
        
        coup->getHands(h1, h2);
        FiberBinder pos;
        Hand * chosen = RNG.flip() ? h1 : h2;
        
        pos = FiberBinder(chosen->fiber(), chosen->abscissa());
#ifdef MULTI_LATTICE
        set_lat_id(chosen->get_fiberlat_id());
#endif
        if (attachmentAllowed(pos))
            attach(pos);
        else
            nextAttach = RNG.exponential();
    }
    
   
}

