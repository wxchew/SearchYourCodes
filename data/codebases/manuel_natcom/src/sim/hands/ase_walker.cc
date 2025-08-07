// Cytosim was created by Francois Nedelec. Copyright 2007-2017 EMBL.
#include "digit.h"
#include "ase_walker.h"
#include "ase_walker_prop.h"
#include "glossary.h"
#include "lattice.h"
#include "simul.h"
extern Random RNG;


//------------------------------------------------------------------------------

AseWalker::AseWalker(AseWalkerProp const* p, HandMonitor* h)
: Digit(p,h), nextStep(0), prop(p)
{
    
}


//------------------------------------------------------------------------------

void AseWalker::attach(FiberBinder const& fb)
{
    Digit::attach(fb);
    nextStep = RNG.exponential();
#ifdef MULTI_LATTICE
    if (prop->redistribution_rate>0)
        nextRedist = RNG.exponential();
#endif
}


void AseWalker::stepUnloaded()
{
    assert_true( attached() );
    if (!prop->is_gillespie)
    {
        // If both positions are occupied, skip
        int check_m = checkM();
        int check_p = checkP();
        if (!(check_m==1&&check_p==1))
        {
            
            nextStep   -= (prop->diff_rate_dt*(check_m!=1) + prop->diff_rate_dt*(check_p!=1));

            if ( nextStep <= 0 )
    //        if (prop->diff2_rate_dt>RNG.preal())
            {
                assert_true( attached() );
                bool move_forward;

                if (check_m==1)
                    move_forward = true;
                else if (check_p==1)
                    move_forward = false;
                else
                    move_forward = RNG.flip();
                
                // See wether to move to the left or to the right by diffussion
                if (move_forward)
                {
                    // The site is normal and free, move forward
                    if (check_p==0)
                        resite(mSite+1);
                    // The plus end is reached
                    else if ( testDangling() )
                    {
                        detach();
                        return;
                    }
                }
                else
                {
                    // The site is normal and free, move forward
                    if (check_m==0)
                        resite(mSite-1);
                    // The plus end is reached
                    else if ( testDangling() )
                    {
                        detach();
                        return;
                    }
                }
                nextStep = RNG.exponential();
            }
        }
        // We try the redistribution here
#ifdef MULTI_LATTICE
        if (prop->redistribution_rate>0)
        {
            nextRedist-= prop->redistribution_rate_dt;
            if (nextRedist<0)
            {
                unsigned int old_lat_id = lat_id;
                // Get a random value and try to move there. 1 out of 4 times it will get the lattice value of where it is, so effectively only switches 3/4 times
                set_lat_id(RNG.pint(4)+1);
                
                
                if (!mLattice->vacant(mSite,lat_id))
                    // Restore the original value
                    lat_id = old_lat_id;
                else
                {
                    // Move to the adjacent lattice
                    mLattice->dec(mSite, old_lat_id);
                    mLattice->inc(mSite, lat_id);
                }
                nextRedist=RNG.exponential();
            }
        }
#endif
    }
    testDetachment();
}

void AseWalker::calc_propensities(Vector const& force, real& p_plus, real& p_min)
{

    // From Hannabuss et al. 2019:
    // step_size_kT = a/kT
    // prop->U_step_kT_2 = (stiffness*a^2)/(2kT)
    // dG_base = f*a/kT
    
    real dG_base = (force * dirFiber())*prop->step_size_kT;
    
    real dG = 0;
    if (p_plus>0)
    {
        dG = dG_base-prop->U_step_kT_2;
        // To avoid zero / zero
        if (dG==0)
            p_plus=1;
        else
            p_plus = dG/(1-exp(-dG));
    }
    
    if (p_min>0)
    {
        dG = -dG_base-prop->U_step_kT_2;
        // To avoid zero / zero
        if (dG==0)
            p_min=1;
        else
            p_min = dG/(1-exp(-dG));
    }
}

void AseWalker::stepLoaded(Vector const& force)
{
    assert_true( attached() );
    
    if (!prop->is_gillespie)
    {
        // If both positions are occupied, skip
        int check_m = checkM();
        int check_p = checkP();
        if (!(check_m==1&&check_p==1))
        {
//            if (check_m==1||check_p==1)
//                std::cout <<"reached" <<std::endl;
//            else
//                std::cout <<"not reached" <<std::endl;
            
            // calculate displacement, dependent on the load along the desired direction of displacement
            real p_plus = 1*(check_p!=1);
            real p_min = 1*(check_m!=1);

            calc_propensities(force, p_plus, p_min);

            nextStep   -= (p_plus + p_min)*prop->diff_rate_dt;
            
            // I changed this from while to if because otherwise it would move several times when the force is high,
            // otherwise it should update the force every time?
            // This was leading to a lot of steps per dt I guess.
            
    //        if ( (p_plus + p_min)*prop->diff_rate_dt > RNG.pfloat())
            if (nextStep<0)
            {
                bool move_forward;
                if (check_m==1)
                    move_forward = true;
                else if (check_p==1)
                    move_forward = false;
                else
                    move_forward = p_plus/(p_plus+p_min)>RNG.preal();
                
                // See wether to move to the left or to the right by diffussion
                if (move_forward)
                {
                    // The site is normal and free, move forward
                    if (check_p==0)
                    {
                        resite(mSite+1);
                    }
                    // The plus end is reached
                    else if ( testDangling() )
                    {
                        detach();
                        return;
                    }
                }
                else
                {
                    // The site is normal and free, move forward
                    if (check_m==0)
                        resite(mSite-1);
                    // The plus end is reached
                    else if ( testDangling() )
                    {
                        detach();
                        return;
                    }
                }
                nextStep = RNG.exponential();
            }
        }
    }
    if ( prop->unbinding_force_inv > 0 )
        testKramersDetachment(force.norm());
    else
        testDetachment();
}


/* ADDED this is a copy paste from the rescuer activity, it might not be the cleanest way to do 
 this, and maybe it should be added in the digit class instead of here */

/**
 Warning:
 This will only work if the time step is small such that only one Hand is
 affected at any time step by the shrinkage of the Fiber.
 Otherwise, the order in which the Hand are considered is random,
 and a distal Hand might be detached, even if the Fiber is rescued prior to this.
 
 This condition should be satisfied for Microtubule systems, since:
 - Shrinking speed ~ -0.1 um/second
 - time_step = 0.010 seconds
 -> shrinkage = 1 nm / time_step
 which would work for the density of 13 binding sites / 8 nm of microtubule.
 */
void AseWalker::handleDisassemblyM()
{
    assert_true( attached() );
    
    if ( RNG.test(prop->rescue_prob) )
    {
        Fiber * fib = fiber();
        assert_true( fbAbs < fbFiber->abscissaM() );
        // induce rescue:
        fib->setDynamicStateM(STATE_GREEN);
        // increase MT length to cover position of Hand
        fib->growM(fiber()->abscissaM()-fbAbs);
        // update all Binders immediately:
        fib->updateBinders();
    }
    else
    {
        // Maybe at some point implement a sweep M? In any case no shrinking minus ends are relevant so far
        Digit::handleDisassemblyM();
    }
}

/**
 Warning:
 This will only work if the time step is small such that only one Hand is
 affected at any time step by the shrinkage of the Fiber.
 Otherwise, the order in which the Hand are considered is random,
 and a distal Hand might be detached, even if the Fiber is rescued prior to this.
 
 This condition should be satisfied for Microtubule systems, since:
 - Shrinking speed ~ -0.1 um/second
 - time_step = 0.010 seconds
 -> shrinkage = 1 nm / time_step
 which would work for the density of 13 binding sites / 8 nm of microtubule.
 */
void AseWalker::handleDisassemblyP()
{
    assert_true( attached() );
    
    // If it rescues, we are good
    if ( RNG.test(prop->rescue_prob) )
    {
        // TODO check whether this works correctly
        Fiber * fib = fiber();
        assert_true( fbAbs > fbFiber->abscissaP() );
        // induce rescue:
        fib->setDynamicStateP(STATE_GREEN);
        // increase MT length to cover position of Hand
        fib->growP(fbAbs-fiber()->abscissaP());
        // update all Binders immediately:
        fib->updateBinders();
    }
    // If it always remained attached it would be problematic because it would rescue too much.
    // I dont think this should be mediated by normal detachment, since Ase1 is suposed to have very strong interactions,
    // and the in vitro observations showed that there was detachment some times.
    else if ( prop->hold_shrinking_end && RNG.pfloat()>prop->hold_shrinking_prob )
    {
        if ( jumpToEnd(PLUS_END) )
            detach();
    }
    else
        detach();
}



real AseWalker::propensUnloaded()
{
    assert_true( attached() );
    return prop->diff_rate_2;
}

real AseWalker::propensLoaded(Vector const& force)
{
    assert_true( attached() );
    
    // calculate displacement, dependent on the load along the desired direction of displacement
    real p_plus = 1;
    real p_min = 1;
    calc_propensities(force, p_plus, p_min);
    return (p_plus+p_min)*prop->diff_rate;
}

int AseWalker::gillestepUnloaded()
{
    // This is copied from step(), probably should figure out a cleaner way of writing this
    assert_true( attached() );
    if ( RNG.flip() ? stepP() == 2 : stepM() == 2 )
    {
        // we have encountered the tip of the fiber
        if ( testDangling() )
        {
            detach();
            return 1;
        }
    }
    return 0;
}

int AseWalker::gillestepLoaded(Vector const & force)
{
    real p_plus = 1;
    real p_min = 1;
    calc_propensities(force, p_plus, p_min);

    real isforward = p_plus/(p_plus+p_min)>RNG.preal();
    if ( isforward ? stepP() == 2 : stepM() == 2 )
    {
        // we have encountered the tip of the fiber
        if ( testDangling() )
        {
            detach();
            return 1;
        }
    }
    return 0;
}

bool AseWalker::needs_sweep()
{
    // Basically checks whether the digit is in a postion bigger than the maximum and returns true if it is
    if ( mSite >= int(fiber()->abscissaP() / prop->step_size) )
        return true;
    else
        return false;
}
void AseWalker::sweep()
{
    // Here there should be a creation of a list of couples that could be potentially swept and send
    // the list as an argument, however, for the purpose of doing this fast, this should work, and it
    // should be correct
    int what = stepM();
    Node * other;
    Hand * ha;
    // In this case, sweep the obstacle
    if (what==1)
    {
        other = next();
        ha = static_cast<Hand*>(other);
        // Iterate until you find one ase_hand or until the end
        // TODO THIS IS INCORRECT I HAVE TO ASK FRANCOIS 
        while (other && ha->prop->name()!="ase_hand")
        {
            other = next();
            ha = static_cast<Hand*>(other);
        }
    }
    else
    {
        other = prev();
        ha = static_cast<Hand*>(other);
        // Iterate until you find one ase_hand or until the end
        while (other && ha->prop->name()!="ase_hand")
        {
            other = prev();
            ha = static_cast<Hand*>(other);
        }
        // Minus end reached, detach BOTH because of the sweeping, and do the sweeping of the previous object
        // Here maybe its better that it detaches both? THere is that possibility
        if (what==2)
        {    detach();
#if (0)
            ha = otherHand();
            if (ha && ha->attached())
                ha->detach();
#endif
        }
    }
    // ha will only be true if the ase_hand has been found, otherwise it will be the end of the nodelist, and
    // thefore false
    if (ha)
        ha->sweep();
    
}


bool AseWalker::first_sweep()
{
    if ( RNG.test(prop->rescue_prob) )
    {
        // TODO check whether this works correctly
        Fiber * fib = fiber();
        assert_true( fbAbs > fbFiber->abscissaP() );
        // induce rescue:
        fib->setDynamicStateP(STATE_GREEN);
        // increase MT length to cover position of Hand
        fib->growP(fbAbs-fiber()->abscissaP());
        // update all Binders immediately:
        fib->updateBinders();
        return true;
    }
    else
        return false;
}

bool AseWalker::testDangling()
{
#ifdef TRAP_SINGLES
    return !trapped() && RNG.test_not(prop->dangling_chance);
#else
    return RNG.test_not(prop->dangling_chance);
#endif
    
}
