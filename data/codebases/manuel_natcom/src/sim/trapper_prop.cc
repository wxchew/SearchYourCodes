
#include "dim.h"
#include "messages.h"
#include "exceptions.h"
#include "glossary.h"
#include "common.h"
#include "simul_prop.h"
#include "trapper_prop.h"

#include "trapper.h"

#include "trapper_long.h"

void TrapperProp::clear()
{
    CoupleProp::clear();
    untrapping_force = INFINITY;
    trapping_rate = 0;
    untrapping_rate = 0;
    target_prop = 0;
    trap_stiffness = 0;
    target_prop_name = "";
    trap_other_filament = false;
    trapping_rate_solution = 0;
    
}

void TrapperProp::read(Glossary& glos)
{
    CoupleProp::read(glos);
//    glos.set(target_lat,        "target_lat");
    glos.set(trapping_rate,     "trapping_rate");
    glos.set(untrapping_rate,   "untrapping_rate");
    glos.set(untrapping_force,  "untrapping_force");
#if (TRAP_SINGLES==1)
    glos.set(trap_stiffness,    "trap_stiffness");
#endif
    glos.set(target_prop_name, "target");
    glos.set(trap_other_filament, "trap_other_filament");
    glos.set(trapping_rate_solution, "trapping_rate_solution");
    
}

void TrapperProp::complete(Simul const* sim)
{
    CoupleProp::complete(sim);
    inv_untrapping_force = 1/untrapping_force;
    trapping_rate_dt = trapping_rate*sim->prop->time_step/sim->prop->handmonitor_pace;
    untrapping_rate_dt = untrapping_rate*sim->prop->time_step/sim->prop->handmonitor_pace;
    if ( target_prop_name.empty() )
        throw InvalidParameter("must define the target class");
    HandProp * int_prop = static_cast<HandProp*>(sim->properties.find_or_die("hand", target_prop_name));
    // In principle only digits should be accepted
    if (!int_prop->lat_val)
        throw InvalidParameter("Target class should be a digit");
    target_prop = static_cast<HandProp*>(int_prop);
    target_lat = target_prop->lat_val;
#if (TRAP_SINGLES==2)
    trap_stiffness = stiffness*2./3.;
#endif
    trapping_rate_solution_dt = trapping_rate_solution*sim->prop->time_step/sim->prop->handmonitor_pace;
}


Couple * TrapperProp::newCouple(Glossary*) const
{
    //std::clog << "CrosslinkProp::newCouple" << std::endl;
    if ( length > 0 )
        return new TrapperLong(this);
    else
        return new Trapper(this);
}

void TrapperProp::write_values(std::ostream & os) const
{
    CoupleProp::write_values(os);
    write_value(os, "trapping_rate",            trapping_rate);
    write_value(os, "untrapping_rate",          untrapping_rate);
    write_value(os, "untrapping_force",         untrapping_force);
#if (TRAP_SINGLES==1)
    write_value(os, "trap_stiffness",           trap_stiffness);
#endif
    write_value(os, "target",                   target_prop_name);
    write_value(os, "trapping_rate_solution",                   trapping_rate_solution);
}


