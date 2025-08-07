// Cytosim was created by Francois Nedelec. Copyright 2007-2017 EMBL.
#include "dim.h"
#include "slider_prop.h"
#include "exceptions.h"
#include "glossary.h"
#include "common.h"
#include "sim.h"
#include <cmath>
#include "property_list.h"
#include "simul_prop.h"
#include "slider.h"


Hand * SliderProp::newHand(HandMonitor* h) const
{
    return new Slider(this, h);
}


void SliderProp::clear()
{
    HandProp::clear();

    mobility = 0;
    stiffness = -1;
}


void SliderProp::read(Glossary& glos)
{
    HandProp::read(glos);
    
    glos.set(mobility,    "mobility");
    glos.set(stiffness,   "stiffness");
}


void SliderProp::complete(Simul const* sim)
{
    HandProp::complete(sim);

    if ( mobility < 0 )
        throw InvalidParameter("slider:mobility must be >= 0");
    
    if ( sim->prop->strict && mobility <= 0 )
        std::clog << "WARNING: slider `" << name() << "' will not slide because mobility=0\n";

    /*
     Explicit
     */
    
    mobility_dt = sim->prop->time_step/sim->prop->handmonitor_pace * mobility;
    
    if ( stiffness > 0 )
    {
        /*
         We devise here an implicit integration approach, assuming:
         - that all other elements of the simulation are static
         - that the link is Hookean of zero resting length:
         force = stiffness * offset
         However, this is true only if the Slider is part of a Single or a plain Couple.
         This does not hold in particular for any of the non-zero resting length Couple or Single.
         J. Ward found that in this case, the numerical precision is not improved compared to
         the explicit integration above.
         */
        std::clog << "         slider:mobility explicit = " << mobility_dt << std::endl;
        
        mobility_dt = ( 1 - exp( - mobility_dt * stiffness ) ) / stiffness;
        std::clog << "         slider:mobility implicit = " << mobility_dt << std::endl;
    }
}


void SliderProp::checkStiffness(real stiff, real len, real mul, real kT) const
{
    HandProp::checkStiffness(stiff, len, mul, kT);
    
    /*
     Estimate numerical stability from mobility and stiffness
     */
    real e = mobility_dt * stiff * mul;
    if ( e > 0.5 )
    {
        std::ostringstream oss;
        oss << "Slider `" << name() << "' may be unstable because:\n";
        oss << PREF << "time_step * mobility * stiffness = " << e << "\n";
        oss << PREF << "reduce mobility or time_step\n";
        //throw InvalidParameter(oss.str());
        std::clog << oss.str();
    }
}


void SliderProp::write_values(std::ostream & os) const
{
    HandProp::write_values(os);
    write_value(os, "mobility",  mobility);
    write_value(os, "stiffness", stiffness);
}

