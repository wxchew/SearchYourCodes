// Cytosim was created by Francois Nedelec. Copyright 2007-2017 EMBL.

#include "dim.h"
#include "sim.h"
#include "exceptions.h"
#include "glossary.h"
#include "common.h"
#include "property_list.h"
#include "simul_prop.h"
#include "digit_prop.h"
#include "digit.h"


Hand * DigitProp::newHand(HandMonitor* h) const
{
    return new Digit(this, h);
}


void DigitProp::clear()
{
    HandProp::clear();
    
    use_lattice       = false;
    step_size         = 0;
    lat_val           = 1;
#ifdef MULTI_LATTICE
    redistribute_on_binding = false;
#endif
}


void DigitProp::read(Glossary& glos)
{
    HandProp::read(glos);
    
    glos.set(step_size,     "step_size");
    glos.set(use_lattice,   "use_lattice");
#ifdef MANU_LATTICE
    glos.set(lat_val, "lat_val");
#endif
#ifdef MULTI_LATTICE
    glos.set(redistribute_on_binding, "redistribute_on_binding");
#endif
}


void DigitProp::complete(Simul const* sim)
{
    HandProp::complete(sim);
    
    if ( step_size <= 0 )
        throw InvalidParameter("Digit:step_size must be defined and > 0");
    
#ifdef DIGITS_GILLESPIE
    // We store the dt, to compare with the firing time and not include in the list if its bigger.
    dt_for_gillespie = sim->prop->time_step;
#endif
#ifdef MANU_LATTICE
#ifdef MULTI_LATTICE
    lat_val_add = 0;
    lat_val_add = (lat_val-1) * 4;
#endif
#endif
    
}


void DigitProp::write_values(std::ostream & os) const
{
    HandProp::write_values(os);
    write_value(os, "step_size", step_size);
    write_value(os, "use_lattice", use_lattice);
}

