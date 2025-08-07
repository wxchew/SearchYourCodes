// Cytosim was created by Francois Nedelec. Copyright 2007-2017 EMBL.
#include "shackle.h"
#include "shackle_prop.h"
#include "fiber_locus.h"
#include "meca.h"


Shackle::Shackle(ShackleProp const* p, Vector const & w)
: Couple(p, w), prop(p)
{
}


Shackle::~Shackle()
{
    prop = 0;
}


/**
 The interaction is slipery on hand1
 */
void Shackle::setInteractions(Meca & meca) const
{
    PointInterpolated pt1 = cHand1->interpolation();
    PointInterpolated pt2 = cHand2->interpolation();

    meca.interSlidingLink(pt1, pt2, prop->stiffness);
}


void Shackle::stepAA()
{
    real dis;
    
    // project the position of cHand2 to set abscissa of cHand1
    real abs = cHand1->fiber()->projectPoint(cHand2->pos(), dis);
    
    //std::clog << "Shackle " << proj.abscissa() - cHand1->abscissa() << std::endl; 
    cHand1->moveTo(abs);
    
    if ( cHand1->attached() )
    {
        Vector f = force();
        cHand1->stepLoaded( f);
        cHand2->stepLoaded(-f);
    }
}
