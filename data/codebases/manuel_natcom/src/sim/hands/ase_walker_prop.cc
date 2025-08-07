// Cytosim was created by Francois Nedelec. Copyright 2007-2017 EMBL.
#include "dim.h"
#include "sim.h"
#include "exceptions.h"
#include "glossary.h"
#include "simul_prop.h"
#include "ase_walker_prop.h"
#include "ase_walker.h"


Hand * AseWalkerProp::newHand(HandMonitor* h) const
{
    return new AseWalker(this, h);
}


void AseWalkerProp::clear()
{
    DigitProp::clear();
    diffusion = 0;
    dangling_chance = 0;
    rescue_prob = 0;
#ifdef MULTI_LATTICE
    redistribution_rate = 0;
#endif
    hold_shrinking_prob = 0;
}


void AseWalkerProp::read(Glossary& glos)
{
    DigitProp::read(glos);
    glos.set(diffusion, "diffusion");
    glos.set(dangling_chance, "dangling_chance");
    glos.set(rescue_prob, "rescue_prob");
    glos.set(hold_shrinking_prob, "hold_shrinking_prob");
#ifdef BACKWARD_COMPATIBILITY
    if ( glos.set(hold_growing_end,  "hold_fiber") )
        MSG.warning("you should use hand:hold_growing_end instead of hand:hold_fiber\n");
#endif
#ifdef MULTI_LATTICE
    glos.set(redistribution_rate, "redistribution_rate");
#endif
}


void AseWalkerProp::complete(Simul const* sim)
{
    DigitProp::complete(sim);
    diff_rate         = diffusion / (step_size*step_size);
    diff_rate_2       = diff_rate * 2;
    diff_rate_dt      = diff_rate * sim->prop->time_step/sim->prop->handmonitor_pace;
    diff2_rate_dt     = diff_rate_dt*2;
    step_size_kT      = step_size/sim->prop->kT;
    
#ifdef MULTI_LATTICE
    redistribution_rate_dt = redistribution_rate * sim->prop->time_step/sim->prop->handmonitor_pace;
#endif
}

void AseWalkerProp::complete_from_couple(Simul const* sim,real stiffness)
{
    U_step_kT_2 = (stiffness*step_size*step_size)/2./sim->prop->kT;
    exp_U_step_kT_2 = exp(U_step_kT_2);
}

void AseWalkerProp::checkStiffness(real stiff, real len, real mul, real kT) const
{
    DigitProp::checkStiffness(stiff, len, mul, kT);
}


void AseWalkerProp::write_values(std::ostream & os) const
{
    DigitProp::write_values(os);
    write_value(os, "diffusion",  diffusion);
    write_value(os, "rescue_prob",  rescue_prob);
    
}

