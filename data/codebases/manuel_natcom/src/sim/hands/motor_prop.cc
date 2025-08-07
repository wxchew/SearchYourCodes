// Cytosim was created by Francois Nedelec. Copyright 2007-2017 EMBL.

#include "dim.h"
#include "messages.h"
#include "exceptions.h"
#include "glossary.h"
#include "common.h"
#include "property_list.h"
#include "simul_prop.h"
#include "motor_prop.h"
#include "motor.h"


Hand * MotorProp::newHand(HandMonitor* h) const
{
    return new Motor(this, h);
}


void MotorProp::clear()
{
    HandProp::clear();
    
    stall_force    = 0;
    unloaded_speed = 0;
    limit_speed    = true;
    
    var_speed_dt = 0;
    set_speed_dt = 0;
    abs_speed_dt = 0;
}


void MotorProp::read(Glossary& glos)
{
    HandProp::read(glos);
    
    glos.set(stall_force,    "stall_force")    || glos.set(stall_force,    "motor_force");
    glos.set(unloaded_speed, "unloaded_speed") || glos.set(unloaded_speed, "motor_speed");
#ifdef BACKWARD_COMPATIBILITY
    glos.set(unloaded_speed,  "max_speed");
#endif
    glos.set(limit_speed,     "limit_speed");
}


void MotorProp::complete(Simul const* sim)
{
    HandProp::complete(sim);
    
    if ( sim->prop->strict && stall_force <= 0 )
        throw InvalidParameter("motor:stall_force must be > 0");

    set_speed_dt = sim->prop->time_step/sim->prop->handmonitor_pace * unloaded_speed;
    abs_speed_dt = fabs(set_speed_dt);
    var_speed_dt = abs_speed_dt / stall_force;
    
    // The limits for a displacement in one time_step apply if ( limit_speed = true )
    if ( unloaded_speed > 0 )
    {
        min_dabs = 0;
        max_dabs = 2 * sim->prop->time_step/sim->prop->handmonitor_pace * unloaded_speed;
    }
    else
    {
        min_dabs = 2 * sim->prop->time_step/sim->prop->handmonitor_pace * unloaded_speed;
        max_dabs = 0;
    }
}



void MotorProp::checkStiffness(real stiff, real len, real mul, real kT) const
{
    HandProp::checkStiffness(stiff, len, mul, kT);
    
    /*
     Compare mobility with stiffness: this can induce instability
     */
    real ef = abs_speed_dt * stiff * mul / stall_force;
    if ( unloaded_speed  &&  ef > 0.5 )
    {
        std::ostringstream oss;
        oss << "motor `" << name() << "' may be unstable:";
        oss << PREF << "time_step * stiffness * unloaded_speed / stall_force = " << ef << "\n";
        oss << PREF << "reduce time_step\n";
        MSG.warning(oss.str());
        //throw InvalidParameter(oss.str());
    }
    
    /* 
    Compare the energy in a link due to the equipartition theorem
     to the maximum force that the motor can sustain before detaching:
     1/2 kT * DIM  <<  1/2 stiffness x^2 ~ 1/2 force^2 / stiffness;
     */
    if ( sqrt( DIM * kT * stiff ) > stall_force )
    {
        std::ostringstream oss;
        oss << "The stall force of `" << name() << "' is too small:\n";
        oss << PREF << "DIM * kT * stiffness > stall_force\n";
        oss << PREF << "reduce stiffness or increase stall_force\n";
        MSG.warning(oss.str());
    }
    
    /*
     Compare the force created by traveling during the time 1/unbinding_rate,
     and compare to stall_force. This is limit the efficiency of the motor.
     */
    ef = fabs( stiff * unloaded_speed / ( unbinding_rate * stall_force ));
    if ( unbinding_rate && unloaded_speed  &&  ef < 1 )
    {
        std::ostringstream oss;
        oss << "The efficiency of `" << name() << "' is low because\n";
        oss << PREF << "stiffness * unloaded_speed / unbinding_rate << stall_force\n";
        oss << PREF << "ratio = " << ef << "\n";
        MSG.warning(oss.str());
    }
    
    /*
     Compare detachment rate at stall-force, with detachment rate at rest
     */
    if ( exp( stall_force * unbinding_force_inv ) > 100 )
        MSG.warning("Hand:exp( stall_force / unbinding_force ) is greater than 100\n");
}



void MotorProp::write_values(std::ostream & os) const
{
    HandProp::write_values(os);
    write_value(os, "stall_force",    stall_force);
    write_value(os, "unloaded_speed", unloaded_speed);
    write_value(os, "limit_speed",    limit_speed);
}

