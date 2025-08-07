// Cytosim was created by Francois Nedelec. Copyright 2007-2017 EMBL.

#include "dim.h"
#include "exceptions.h"
#include "glossary.h"
#include "common.h"
#include "property_list.h"
#include "simul_prop.h"
#include "chewer_prop.h"
#include "chewer.h"


Hand * ChewerProp::newHand(HandMonitor* h) const
{
    return new Chewer(this, h);
}


void ChewerProp::clear()
{
    HandProp::clear();

    chewing_speed = 0;
    diffusion     = 0;
}


void ChewerProp::read(Glossary& glos)
{
    HandProp::read(glos);
    
    glos.set(chewing_speed, "chewing_rate");
    glos.set(chewing_speed, "chewing_speed");
    glos.set(diffusion,     "diffusion");
}


void ChewerProp::complete(Simul const* sim)
{
    HandProp::complete(sim);
    
    if ( chewing_speed < 0 )
        throw InvalidParameter("chewer:chewing_speed must be >= 0");

    chewing_speed_dt = chewing_speed * sim->prop->time_step/sim->prop->handmonitor_pace;
    
    if ( diffusion < 0 )
        throw InvalidParameter("chewer:diffusion must be >= 0");

    /*
     We want for one degree of freedom to fulfill `var(dx) = 2 D dt`
     And we use: dx = diffusion_dt * RNG.sreal()
     Since `sreal()` is uniformly distributed, its variance is 1/3,
     and we need `diffusion_dt^2 = 6 D dt`
     */
    diffusion_dt = sqrt( 6.0 * diffusion * sim->prop->time_step/sim->prop->handmonitor_pace );
    
    // use Einstein's relation to get a mobility:
    mobility_dt = diffusion * sim->prop->time_step/sim->prop->handmonitor_pace / sim->prop->kT;
    
    std::clog << " Chewer `" << name() << "' has mobility = " << diffusion / sim->prop->kT << "\n";
}


void ChewerProp::checkStiffness(real stiff, real len, real mul, real kT) const
{
    HandProp::checkStiffness(stiff, len, mul, kT);
    
    /*
     Compare mobility with stiffness: this can induce instability
     */
    real a = mobility_dt * stiff * mul;
    if ( a > 1.0 )
    {
        std::ostringstream oss;
        oss << "chewer `" << name() << "' may be unstable:\n";
        oss << PREF << "mobility = " << diffusion / kT << "\n";
        oss << PREF << "mobility * stiffness * time_step = " << a << "\n";
        oss << PREF << "> reduce time_step\n";
        //MSG.warning(oss.str());
        throw InvalidParameter(oss.str());
    }
    
}


void ChewerProp::write_values(std::ostream & os) const
{
    HandProp::write_values(os);
    write_value(os, "chewing_speed", chewing_speed);
    write_value(os, "diffusion",     diffusion);
}

