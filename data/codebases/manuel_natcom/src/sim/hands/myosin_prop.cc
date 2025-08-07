// Cytosim was created by Francois Nedelec. Copyright 2007-2017 EMBL.

#include "myosin.h"
#include "myosin_prop.h"
#include "exceptions.h"
#include "glossary.h"
#include "simul_prop.h"


Hand * MyosinProp::newHand(HandMonitor* h) const
{
    return new Myosin(this, h);
}



void MyosinProp::clear()
{
    DigitProp::clear();

    stall_force    = 0;
    unloaded_speed = 0;
}


void MyosinProp::read(Glossary& glos)
{
    DigitProp::read(glos);
    
    glos.set(stall_force,    "stall_force")    || glos.set(stall_force,    "motor_force");
    glos.set(unloaded_speed, "unloaded_speed") || glos.set(unloaded_speed, "motor_speed");
#ifdef BACKWARD_COMPATIBILITY
    glos.set(unloaded_speed, "max_speed");
#endif
}


void MyosinProp::complete(Simul const* sim)
{
    DigitProp::complete(sim);
   
    if ( sim->prop->strict && stall_force <= 0 )
        throw InvalidParameter("myosin:stall_force must be > 0");
    
    if ( unloaded_speed < 0 )
        throw InvalidParameter("myosin:unloaded_speed must be >= 0");

    stepping_rate     = fabs(unloaded_speed) / step_size;
    stepping_rate_dt  = sim->prop->time_step/sim->prop->handmonitor_pace * stepping_rate;
    var_rate_dt       = ( unloaded_speed >= 0 ? 1 : -1 ) * stepping_rate_dt / stall_force;
}



void MyosinProp::write_values(std::ostream & os) const
{
    DigitProp::write_values(os);
    write_value(os, "stall_force",    stall_force);
    write_value(os, "unloaded_speed", unloaded_speed);
}

