

#if (0)
// Cytosim was created by Francois Nedelec. Copyright 2007-2017 EMBL.

#include "trapper.h"
#include "trapper_prop.h"
#include "exceptions.h"
#include "random.h"
#include "modulo.h"
#include "meca.h"
#include "couple_set.h"
#include "simul.h"
#include "sim.h"


extern Random RNG;
extern Modulo const* modulo;

//------------------------------------------------------------------------------
Trapper::Trapper(TrapperProp const* p, Vector const& w)
: Couple(p, w), prop(p)
{
}


Trapper::~Trapper()
{
    Couple::~Couple();
}

Vector Trapper::forceTrap()
{
    Vector d = trappedOtherHandPos() - trappedHandPos();
    //correct for periodic space:
    if ( modulo )
        modulo->fold(d);
    return prop->trap_stiffness * d;
    
}
Vector Trapper::forceTrap1()
{
    Vector d = cHand1->trappedOtherHandPos() - cHand1->pos();
    //correct for periodic space:
    if ( modulo )
        modulo->fold(d);
    return prop->trap_stiffness * d;
    
}
Vector Trapper::forceTrap2()
{
    Vector d = cHand2->trappedOtherHandPos() - cHand2->pos();
    //correct for periodic space:
    if ( modulo )
        modulo->fold(d);
    return prop->trap_stiffness * d;
    
}


bool Trapper::try2Trap(Hand * cHand)
{
    // See if in thd lattice site of the hand there is another hand wiht lat_val prop->target_lat (the target single should move on that lattice)
    if (cHand->find_lattice_neighbour(prop->target_lat))
    {
        // Returns the hand, but this hand might be trapped by another couple
        FiberBinder * fb = cHand->fiber()->findHand(prop->target_lat,cHand->lattice_site());
        Hand * h = static_cast<Hand*>(fb);
        if (h && !h->trapped())
        {
            h->trap(cHand);
            return true;
        }
    }
    return false;
}

#if (1)
void Trapper::step_untrapped()
{
    next_trap-=prop->trapping_rate_dt;
    if (next_trap<0)
    {
        // Try to attach one of the two heads of the couple
        if (RNG.flip() ? try2Trap(cHand1,cHand2) : try2Trap(cHand2,cHand1))
            // Reset the untrapping
            next_untrap = RNG.exponential();
        else
            // If the trapping attempt was unsuccesful, assign a new random number
            next_trap = RNG.exponential();
    }
}

void Trapper::step_trapped(Vector f_trap)
{
    real force = f_trap.norm();
    next_untrap-=prop->untrapping_rate_dt* exp(force*prop->inv_untrapping_force);
    if (next_untrap<0)
    {
        untrap(true);
        next_trap = RNG.exponential();
    }
}
#else
void Trapper::step_untrapped()
{
    //    next_trap-=prop->trapping_rate_dt;
    //    if (next_trap<0)
    if(RNG.test(prop->trapping_rate_dt))
    {
        // Try to attach one of the two heads of the couple
        if (RNG.flip() ? try2Trap(cHand1) : try2Trap(cHand2))
            // Reset the untrapping
            next_untrap = RNG.exponential();
        else
            // If the trapping attempt was unsuccesful, assign a new random number
            next_trap = RNG.exponential();
    }
}

void Trapper::step_trapped(Vector f_trap)
{
    //    real force = f_trap.norm();
    //    if (prop->inv_untrapping_force>0)
    //        next_untrap-=prop->untrapping_rate_dt* exp(force*prop->inv_untrapping_force);
    //    else
    //        next_untrap-=prop->untrapping_rate_dt;
    //
    //    if (next_untrap<0)
    if (RNG.test(prop->untrapping_rate_dt))
        untrap(true);
}
#endif
void Trapper::untrap(bool first_call)
{
    assert_true(trapped);
    
    if (cHand1->trapped())
    {
        // If it is the first call, we have to also untrap the partner
        if (first_call)
            cHand1->trapped_haMon->untrap(false);
        cHand1->untrap();
    }
    else
    {
        if (first_call)
            cHand2->trapped_haMon->untrap(false);
        cHand2->untrap();
    }
}

void Trapper::stepAA()
{
    // THE NORMAL STEP ---------------------------------------------
    
    Vector f = force();
    Vector f_trap;
    if (cHand1->trapped())
    {
        f_trap = forceTrap1();
        // Any of these steps can lead to unbinding of the couple, which untraps the single, so we run the trapping step first.
        // This eventually could be called directly from the singleset of
        // trapped singles, but it will be faster from here so the force is calculated
        // only once. Although it is a bit more ugly.
        cHand1->stepTrapped(-f_trap);
        cHand1->stepLoaded(f+f_trap);
        cHand2->stepLoaded(-f);
        
        // If the hand was untrapped during this, we dont try to trap it again
        if (cHand1->trapped())
            step_trapped(f_trap);
    }
    
    else if (cHand2->trapped())
    {
        f_trap = forceTrap2();
        cHand2->stepTrapped(-f_trap);
        cHand1->stepLoaded(f);
        cHand2->stepLoaded(-f+f_trap);
        if (cHand2->trapped())
            step_trapped(f_trap);
    }
    else
    {
        // The usual step
        cHand1->stepLoaded( f);
        cHand2->stepLoaded(-f);
        // If none of them unbind
        if (linking())
            step_untrapped();
    }
}


void Trapper::afterAttachment(Hand const* h)
{
    
    Couple::afterAttachment(h);
    // Initialize the counter for capturing a single
    if (linking()) {
        next_trap = RNG.exponential();
    }
}

void Trapper::setInteractions(Meca & meca) const
{
    Couple::setInteractions(meca);
    if (prop->trap_other_filament)
    {
        if (cHand1->trapped())
            meca.interLink(cHand1->interpolation(), cHand1->trapped_hand()->interpolation(), prop->stiffness);
        else if (cHand2->trapped())
            meca.interLink(cHand2->interpolation(), cHand2->trapped_hand()->interpolation(), prop->stiffness);
    }
}

void Trapper::beforeDetachment(Hand const* h)
{
    if (trapped())
        untrap(true);
    Couple::beforeDetachment(h);
}

void Trapper::write(Outputter& out) const
{
    Couple::write(out);
    
    if (cHand1->trapped())
        cHand1->trapped_haMon->writeReference(out);
    else
        Object::writeNullReference(out);
    
    
    if (cHand2->trapped())
        cHand2->trapped_haMon->writeReference(out);
    else
        Object::writeNullReference(out);
}

void Trapper::read(Inputter & in, Simul& sim, Tag tag)
{
    Couple::read(in, sim, tag);
    
    // First hand
    Object * w = sim.readReference(in, tag);
    if (w) cHand1->trapped_haMon = static_cast<Single*>(w);
    else cHand1->untrap();
    // Second hand
    w = sim.readReference(in, tag);
    if (w) cHand2->trapped_haMon = static_cast<Single*>(w);
    else cHand2->untrap();
}

Hand * Trapper::trappedHand() const
{
    assert_true(trapped());
    if (cHand1->trapped())
        return cHand1;
    else
        return cHand2;
}
Vector Trapper::trappedHandPos() const
{
    assert_true(trapped());
    return trappedHand()->pos();
}
Vector Trapper::trappedOtherHandPos() const
{
    return trappedOtherHand()->pos();
}
Hand * Trapper::trappedOtherHand() const
{
    return trappedHand()->trapped_haMon->trappedHand();
}


void Trapper::writeReference(Outputter& out) const
{
    Object::writeReference(out,tag());
}


#endif