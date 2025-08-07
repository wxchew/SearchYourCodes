


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
    trapped_haMon = 0;
}


Trapper::~Trapper()
{
    Couple::~Couple();
}

real Trapper::get_stiffness() const
{
    return (trapped() && trappedHand()->attached()) ? prop->trap_stiffness : prop->stiffness;
}

Vector Trapper::force() const
{
    real stiffness = get_stiffness();
    Vector d = cHand2->pos() - cHand1->pos();
    
    //correct for periodic space:
    if ( modulo )
        modulo->fold(d);
    
    return stiffness * d;

}

Vector Trapper::forceTrap1() const
{
    real stiffness = get_stiffness();
    // Single will return the hand that is engaged
    Vector d = trappedHand()->pos() - cHand1->pos();
    //correct for periodic space:
    if ( modulo )
        modulo->fold(d);
    return stiffness * d;
    
}
Vector Trapper::forceTrap2() const
{
    real stiffness = get_stiffness();
    Vector d = trappedHand()->pos() - cHand2->pos();
    //correct for periodic space:
    if ( modulo )
        modulo->fold(d);
    return stiffness * d;
    
}
bool Trapper::try2Trap(Hand * cHand)
{
    // See if in the lattice site of the hand there is another hand wiht lat_val prop->target_lat (the target single should move on that lattice)
#ifdef MULTI_LATTICE
    int target = (prop->target_lat-1)*4 + cHand->get_fiberlat_id();
    if (cHand->find_lattice_neighbour(target))
#else
    if (cHand->find_lattice_neighbour(prop->target_lat))
#endif
    {
        // Returns the hand, but this hand might be trapped by another couple
        
#ifdef MULTI_LATTICE
        FiberBinder * fb = cHand->fiber()->findHand(prop->target_lat,cHand->lattice_site(),cHand->get_fiberlat_id());
        assert_true(fb);
#else
        FiberBinder * fb = cHand->fiber()->findHand(prop->target_lat,cHand->lattice_site());
#endif

        Hand * h = static_cast<Hand*>(fb);
        
        if (h && !h->trapped())
        {
            trap(h->haMonitor);
            return true;
        }
    }
    return false;
}

void Trapper::trap(HandMonitor * haMon)
{
    trapped_haMon = haMon;
    haMon->trap(this);
    next_untrap = RNG.exponential();
}

void Trapper::untrap()
{
    assert_true(trapped());
    // First we unlink the single
    trapped_haMon->untrap();
    trapped_haMon = 0;
    next_trap = RNG.exponential();
}


void Trapper::stepUntrapped(Hand * cHand)
{
    next_trap-=prop->trapping_rate_dt;
    if (next_trap<0)
    {
        if (!try2Trap(cHand))
            next_trap = RNG.exponential();
    }
}


void Trapper::stepTrappedA(Vector f_trap)
{
    trapped_haMon->stepTrappedA(f_trap);
    real force = f_trap.norm();
    next_untrap-=prop->untrapping_rate_dt* exp(force*prop->inv_untrapping_force);
    if (next_untrap<0)
        untrap();
}

void Trapper::stepTrappedF(const FiberGrid& grid, Vector const & pos)
{
    trapped_haMon->stepTrappedF(grid, pos);
    next_untrap-=prop->untrapping_rate_dt;
    if (next_untrap<0)
        untrap();
}
void Trapper::stepTrappedF_AA()
{
    trapped_haMon->stepTrappedF_AA();
    next_untrap-=prop->untrapping_rate_dt;
    if (next_untrap<0)
        untrap();
}


void Trapper::stepFA(const FiberGrid& grid)
{
    //we use cHand2->pos() first, because stepUnloaded() may detach cHand2
    cHand1->stepUnattached(grid, cHand2->pos());
    if (trapped())
    {
        if (trappedHand()->attached())
        {
            Vector f_2m = forceTrap2();
            // We do the single before, since the couple could unbind, which results in untrapping if there is only one hand bound
            stepTrappedA(-f_2m);
            cHand2->stepLoaded(f_2m);
        }
        else
        {
            stepTrappedF(grid, cHand2->pos());
            cHand2->stepUnloaded();

        }
    }
    else
    {
        stepUntrapped(cHand2);
        cHand2->stepUnloaded();
    }

}

void Trapper::stepAF(const FiberGrid& grid)
{
    //we use cHand1->pos() first, because stepUnloaded() may detach cHand1
    cHand2->stepUnattached(grid, cHand1->pos());
    if (trapped())
    {
        if (trappedHand()->attached())
        {            
            Vector f_1m = forceTrap1();
            // We do the single before, since the couple could unbind, which results in untrapping if there is only one hand bound
            stepTrappedA(-f_1m);
            cHand1->stepLoaded(f_1m);
        }
        else
        {
            stepTrappedF(grid,cHand1->pos());
            cHand1->stepUnloaded();
        }
    }
    else
    {
        stepUntrapped(cHand1);
        cHand1->stepUnloaded();
    }

}

void Trapper::stepAA()
{
    if (trapped())
    {
        if (trappedHand()->attached())
        {
            Vector f_12 = force();
            Vector f_1m = forceTrap1();
            Vector f_2m = forceTrap2();
            
            // In the trapper, we control trapping/untrapping
            stepTrappedA(-f_1m - f_2m);
            
            cHand1->stepLoaded(f_12+f_1m);
            cHand2->stepLoaded(-f_12+f_2m);
        }
        else
        {
            stepTrappedF_AA();
            Couple::stepAA();
        }
    }
    else
    {
        // First these two steps, because they could unbind during the stepAA
        stepUntrapped(cHand1);
        if (!trapped())
            stepUntrapped(cHand2);
        Couple::stepAA();
    }
}


void Trapper::afterAttachment(Hand const* h)
{
    Couple::afterAttachment(h);
    // If it bound from solution, initialize the trapper counter
    if (!linking())
    {
        if (trapped())
            next_untrap = RNG.exponential();
        else
            next_trap = RNG.exponential();
    }
}

void Trapper::setInteractions(Meca & meca) const
{
    if (trapped() && trappedHand()->attached())
    {
        meca.interSimpleTriLink(cHand1->interpolation(),cHand2->interpolation(),trappedHand()->interpolation(),prop->trap_stiffness);
    }
    // If its not trapped/the hand is not attached, do the usual couple interactions
    else
        Couple::setInteractions(meca);
}

void Trapper::setInteractionsAF(Meca & meca) const
{
    if (trapped() && trappedHand()->attached() && (cHand1->fiber()!=trappedHand()->fiber()))
    {
        // Here we use the "normal" stiffness, since only two couples are linked
        meca.interLink(cHand1->interpolation(), trappedHand()->interpolation(), prop->stiffness);
    }
}

void Trapper::setInteractionsFA(Meca & meca) const
{
    if (trapped() && trappedHand()->attached() && (cHand2->fiber()!=trappedHand()->fiber()))
    {
        // Here we use the "normal" stiffness, since only two couples are linked    
        meca.interLink(cHand2->interpolation(), trappedHand()->interpolation(), prop->stiffness);
    }
}

void Trapper::beforeDetachment(Hand const* h)
{

    if (!otherHand(h)->attached() && trapped())
        untrap();
    Couple::beforeDetachment(h);
}

void Trapper::write(Outputter& out) const
{
    Couple::write(out);
    
    if (trapped())
        trapped_haMon->writeReference(out);
    else
        Object::writeNullReference(out);
}

void Trapper::read(Inputter & in, Simul& sim, Tag tag)
{
    Couple::read(in, sim, tag);
    
    Object * w = sim.readReference(in, tag);
    if (w) trapped_haMon = static_cast<Single*>(w);
    else trapped_haMon = 0;
}

void Trapper::writeReference(Outputter& out) const
{
    Object::writeReference(out,tag());
}

// Auxiliary function to not write the same thing a hundred times

Vector3 trapCenterAdd(Hand * h, real & attached_num,real exp_shift)
{
    Vector3 out(0,0,0);
    if (h->attached())
    {
        Vector pos = h->pos();
        attached_num++;
        out.XX+=pos.XX;
        out.YY+=exp_shift;
        
#if (DIM==2)
        out.YY+=pos.YY;
#elif (DIM==3)
        out.YY+=pos.YY;
        out.ZZ+=pos.ZZ;
#endif
    }
    return out;
}

Vector3 Trapper::trapCenter(real exp_shift1,real exp_shift2,real exp_shift3) const
{
    Vector3 out(0,0,0);
    real attached_num = 0;

    out += trapCenterAdd(cHand1, attached_num, exp_shift1);
    out += trapCenterAdd(cHand2, attached_num, exp_shift2);
    out += trapCenterAdd(trappedHand(), attached_num, exp_shift3);
    out /= attached_num;
    return out;
}

Hand * Trapper::trappedHand() const
{
    // Single will return the hand that is engaged
    Hand * h1 = nullptr, * h2 = nullptr;

    trapped_haMon->getHands(h1, h2);
    return h1;
}

void Trapper::checkAll() const
{
    if (cHand1->attached())
        assert_false(fiber1()->frBindersBad());
    if (cHand2->attached())
        assert_false(fiber2()->frBindersBad());
    if (trapped() && trappedHand()->attached()) {
        assert_false(trappedHand()->fiber()->frBindersBad());
    }
    
}

real Trapper::trap_rate() const
{
    return prop->trapping_rate_solution_dt;
}

real Trapper::untrap_rate() const
{
    return prop->untrapping_rate_dt;
}
#ifdef MULTI_LATTICE

// f is the fiber the trapped single is trying to bind to
int Trapper::partnerLattice(Hand * h, Fiber * f) const
{
    if (cHand1->attached())
    {
        if (cHand1->fiber() == f)
            return cHand1->get_fiberlat_id();
        else
        {
            // Here we verify that the couple can bridge these two fibers from that lat_val
            if (cHand1->fiber()->get_lattice_val(f)==cHand1->get_fiberlat_id())
                return f->get_lattice_val(cHand1->fiber());
            else
                return 0;
        }
        
    }
    if (cHand2->attached())
    {
        if (cHand2->fiber() == f)
            return cHand2->get_fiberlat_id();
        else
        {
            if (cHand2->fiber()->get_lattice_val(f)==cHand2->get_fiberlat_id())
                return f->get_lattice_val(cHand2->fiber());
            else
                return 0;
        }
    }
    std::cerr << "Error: Unattached trapped single trying to bind " << std::endl;
    return 0;
}

int Trapper::checkPartnerLattice() const
{
 
    if (trapped())
    {
        Hand * h_trap = trappedHand();
        if (h_trap->attached()&& linking())
        {
            if ((h_trap->fiber()!=cHand1->fiber())&&(h_trap->fiber()!=cHand2->fiber()))
            {
                std::cerr<< "Trapper bound to three different fibers" <<std::endl;
            }
        }
    }
    int result = checkPartnerLattice(cHand1);
    if (result)
        return result;
    result = checkPartnerLattice(cHand2);
    if (result)
        return result;
    return 0;
}

int Trapper::checkPartnerLattice(Hand * cHand) const
{
    if (!cHand->attached())
        return 0;
    if (cHand->fiber()==trappedHand()->fiber())
    {
        if (cHand->get_fiberlat_id()!=trappedHand()->get_fiberlat_id())
            return 1;
    }
    else
    {
        if (trappedHand()->fiber()->get_lattice_val(cHand->fiber())!=trappedHand()->get_fiberlat_id())
            return 2;
    }
    return 0;
}
#endif