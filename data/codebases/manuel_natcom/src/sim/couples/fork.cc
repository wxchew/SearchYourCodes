// Cytosim was created by Francois Nedelec. Copyright 2007-2017 EMBL.

#include "fork.h"
#include "fork_prop.h"
#include "meca.h"


Fork::Fork(ForkProp const* p, Vector const& w)
: Couple(p, w), prop(p)
{
    flip = 1;
}


Fork::~Fork()
{
    prop = 0;
}


void Fork::setInteractions(Meca & meca) const
{
    PointInterpolated pt1 = cHand1->interpolation();
    PointInterpolated pt2 = cHand2->interpolation();
    
    meca.interLink(pt1, pt2, prop->stiffness);
#if ( DIM == 2 )
    // flip the angle to match the current configuration of the bond
    if ( prop->flip )
    {
        if ( cross(pt1.diff(), pt2.diff()) < 0 )
            flip = -1;
        else
            flip = +1;
    }
    
    meca.interTorque2D(pt1, pt2, prop->cosinus, flip * prop->sinus, prop->angular_stiffness);
#else
    meca.interTorque(pt1, pt2, prop->cosinus, prop->sinus, prop->angular_stiffness);
#endif
    // this works for angle == 0:
    //meca.interTorque(pt1, pt2, prop->angular_stiffness);
}



