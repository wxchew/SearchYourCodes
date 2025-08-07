// Cytosim was created by Francois Nedelec. Copyright 2007-2017 EMBL.

#include "dim.h"
#include "sim.h"
#include "exceptions.h"
#include "glossary.h"
#include "key_list.h"
#include "common.h"
#include "property_list.h"
#include "simul_prop.h"
#include "nucleator_prop.h"
#include "nucleator.h"
#include "simul.h"

//------------------------------------------------------------------------------
Hand * NucleatorProp::newHand(HandMonitor* h) const
{
    return new Nucleator(this, h);
}

//------------------------------------------------------------------------------
void NucleatorProp::clear()
{
    HandProp::clear();

    fiber        = "";
    rate         = 0;
    spec         = "";
    specificity  = NUCLEATE_ORIENTATED;
    track_end    = NO_END;
    hold_end     = MINUS_END;
    addictive    = false;
}

//------------------------------------------------------------------------------
void NucleatorProp::read(Glossary& glos)
{
    HandProp::read(glos);
    
    glos.set(rate,   "rate");
    glos.set(fiber,  "fiber");
    glos.set(fiber,  "fibers");
    glos.set(spec,   "spec");
    
    
    glos.set(rate,   "nucleate", 0);
    glos.set(fiber,  "nucleate", 1);
    glos.set(spec,   "nucleate", 2);

#ifdef BACKWARD_COMPATIBILITY
    glos.set(spec,   "nucleation_spec");
#endif
    
    glos.set(addictive,  "addictive");
    
    glos.set(hold_end,   "hold_end",
             KeyList<FiberEnd>("off",         NO_END,
                               "none",        NO_END,
                               "minus_end",   MINUS_END,
                               "plus_end",    PLUS_END));

    glos.set(track_end,  "track_end",
             KeyList<FiberEnd>("off",         NO_END,
                               "none",        NO_END,
                               "minus_end",   MINUS_END,
                               "plus_end",    PLUS_END));
    
    glos.set(specificity, "specificity",
             KeyList<Specificity>("off",          NUCLEATE_ORIENTATED,
                                  "none",         NUCLEATE_ORIENTATED,
                                  "parallel",     NUCLEATE_PARALLEL,
                                  "antiparallel", NUCLEATE_ANTIPARALLEL,
                                  "parallel_if",  NUCLEATE_PARALLEL_IF));
}

//------------------------------------------------------------------------------
void NucleatorProp::complete(Simul const* sim)
{
    HandProp::complete(sim);
    
    simul = const_cast<Simul*>(sim);

    if ( fiber.empty() )
        throw InvalidParameter("hand:nucleate[1] (=fiber) must be specified if activity=nucleate");

    sim->properties.find_or_die("fiber", fiber);
    
    if ( rate < 0 )
        throw InvalidParameter("hand:nucleate (=rate) must be positive");

    if ( track_end && track_end != hold_end )
        throw InvalidParameter("if set, hand:track_end should be equal to hold_end");
        
    rate_dt = rate * sim->prop->time_step/sim->prop->handmonitor_pace;
}


//------------------------------------------------------------------------------

void NucleatorProp::write_values(std::ostream & os) const
{
    HandProp::write_values(os);
    write_value(os, "nucleate",  rate, fiber, "("+spec+")");
    write_value(os, "hold_end",  hold_end);
    write_value(os, "track_end", track_end);
    write_value(os, "addictive", addictive);
}

