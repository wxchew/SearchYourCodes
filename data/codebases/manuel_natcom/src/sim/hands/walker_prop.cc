// Cytosim was created by Francois Nedelec. Copyright 2007-2017 EMBL.
#include "dim.h"
#include "sim.h"
#include "exceptions.h"
#include "glossary.h"
#include "simul_prop.h"
#include "walker_prop.h"
#include "walker.h"


Hand * WalkerProp::newHand(HandMonitor* h) const
{
    return new Walker(this, h);
}


void WalkerProp::clear()
{
    DigitProp::clear();

    stall_force       = 0;
    unloaded_speed    = 0;
    unbinding_chance  = 0;
    dangling_chance   = 0;
    plusEndDirected   = 0;
}


void WalkerProp::read(Glossary& glos)
{
    DigitProp::read(glos);
    
    glos.set(stall_force,    "stall_force")    || glos.set(stall_force,    "motor_force");
    glos.set(unloaded_speed, "unloaded_speed") || glos.set(unloaded_speed, "motor_speed");
#ifdef BACKWARD_COMPATIBILITY
    glos.set(unloaded_speed,    "max_speed");
#endif
    glos.set(unbinding_chance,  "unbinding_chance");
    //alternative syntax:
    glos.set(unbinding_chance,  "unbinding", 2);
    
    glos.set(dangling_chance,   "dangling_chance");
    
    if ( dangling_chance > 0 )
    {
        MSG.warning("defined walker:hold_growing_end=1, because walker:dangling_chance > 0\n");
        hold_growing_end = true;
    }

    
#ifdef BACKWARD_COMPATIBILITY
    if ( glos.set(hold_growing_end,  "hold_fiber") )
        MSG.warning("you should use hand:hold_growing_end instead of hand:hold_fiber\n");
#endif
}


void WalkerProp::complete(Simul const* sim)
{
    DigitProp::complete(sim);
   
    if ( sim->prop->strict && stall_force <= 0 )
        throw InvalidParameter("walker:stall_force must be > 0");
    
    if ( unbinding_chance < 0 )
        throw InvalidParameter("walker:unbinding_chance must be >= 0");

    if ( dangling_chance < 0 )
        throw InvalidParameter("walker:dangling_chance must be >= 0");
    
    stepping_rate     = fabs(unloaded_speed) / step_size;
    stepping_rate_dt  = sim->prop->time_step * stepping_rate/sim->prop->handmonitor_pace;
    var_rate_dt       = ( unloaded_speed >= 0 ? 1 : -1 ) * stepping_rate_dt / stall_force;
    plusEndDirected = unloaded_speed>0;
}


void WalkerProp::checkStiffness(real stiff, real len, real mul, real kT) const
{
    DigitProp::checkStiffness(stiff, len, mul, kT);

#if ( 0 )
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
     Compare the force reached in one step with the stall force
     */
    if ( fabs( step_size * stiff ) > 0.5 * stall_force )
        MSG.warning("attention:  stiffness * digit:step > stall_force / 2\n");
#endif
}



void WalkerProp::write_values(std::ostream & os) const
{
    DigitProp::write_values(os);
    write_value(os, "stall_force",      stall_force);
    write_value(os, "unloaded_speed",   unloaded_speed);
    write_value(os, "unbinding_chance", unbinding_chance);
    write_value(os, "dangling_chance",  dangling_chance);
}

