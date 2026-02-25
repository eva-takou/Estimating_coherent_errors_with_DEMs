import numpy as np

def get_vi_mean(detection_events):

    num_shots = np.shape(detection_events)[0]
    vi        = np.sum(detection_events,axis=0)/num_shots 

    return vi


def get_pij_corr(detection_events,vi_mean,indx1,indx2,pijk,pijkl):
    
    num_shots = np.shape(detection_events)[0]

    v1 = vi_mean[indx1]
    v2 = vi_mean[indx2]

    v1v2 = np.sum((detection_events[:,indx1] & detection_events[:,indx2]))/num_shots

    numer = v1v2-v1*v2

    denom = 1-2*(v1+v2)+4*v1v2

    if (1/4-numer/denom)<0:
        pij=0
        
        print("Bad numer/denom in pij")
        
        return pij
    
    term_to_correct = np.sqrt(1/4-numer/denom)

    for key in pijk.keys():

        inds = []
        for det in key:
            ind = int(det[1:])
            inds.append(ind)
        
        if indx1 in inds and indx2 in inds:
            term_to_correct *= 1/(1-2*pijk[key])

    for key in pijkl.keys():

        inds = []
        for det in key:
            ind = int(det[1:])
            inds.append(ind)
        
        if indx1 in inds and indx2 in inds:
            term_to_correct *= 1/(1-2*pijkl[key])


    pij   = 1/2 - term_to_correct

    if pij<0 or pij>1:
        pij = 1/2+term_to_correct
        if pij<0 or pij>1:
            pij=0

            print("Maybe removed too many higher-order contributions in pij?")
            return pij


    return pij


def get_3pnt_prob(detection_events,vi_mean,indx1,indx2,indx3, pijkl):

    num_shots = np.shape(detection_events)[0]

    v1 = vi_mean[indx1]
    v2 = vi_mean[indx2]
    v3 = vi_mean[indx3]

    v1v2 = np.sum(detection_events[:,indx1] & detection_events[:,indx2])/num_shots
    v1v3 = np.sum(detection_events[:,indx1] & detection_events[:,indx3])/num_shots
    v2v3 = np.sum(detection_events[:,indx2] & detection_events[:,indx3])/num_shots

    v1v2v3 = np.sum(detection_events[:,indx1] & detection_events[:,indx2] & detection_events[:,indx3] )/num_shots 

    denom  = 1 - 2 * (v1+v2) + 4 * v1v2 
    denom *= 1 - 2 * (v1+v3) + 4 * v1v3
    denom *= 1 - 2 * (v2+v3) + 4 * v2v3

    numer  = (1-2*v1)*(1-2*v2)*(1-2*v3)
    numer *= 1 - 2 * (v1+v2+v3) + 4 * (v1v2+v1v3+v2v3) - 8 * v1v2v3
    
    if (numer/denom)<0:
        print("Bad number in pijk for (i,j,k)=",(indx1,indx2,indx3))
         
        return 0

    term_to_correct = 0.5 * (numer/denom)**(1/4)

    # print("Init term in pijk:",term_to_correct, "for ijk=",(indx1,indx2,indx3))

    
    for key in pijkl.keys():
        inds = []
        for det in key:
            ind = int(det[1:])
            inds.append(ind)
        if indx1 in inds and indx2 in inds and indx3 in inds:
            term_to_correct *= 1/(1-2*pijkl[key])
            # print("Correcting the term:",key,"w/ new value:",term_to_correct)

    
    #Need to multiply exclusion factor too: x 1/(1-2*p_ijkl)

    p = 0.5-term_to_correct


    if p<0:
        p=0
        print("Maybe removed too many higher-order contributions in pijk? for ijk=",(indx1,indx2,indx3))
    if np.isnan(p):
        p=0

    return p



def get_4pnt_prob(det_events,vi_mean,indx1,indx2,indx3,indx4):
    
    num_shots = np.shape(det_events)[0]
    v1 = vi_mean[indx1]
    v2 = vi_mean[indx2]
    v3 = vi_mean[indx3]
    v4 = vi_mean[indx4]

    v1v2 = np.sum(det_events[:,indx1] & det_events[:,indx2])/num_shots
    v1v3 = np.sum(det_events[:,indx1] & det_events[:,indx3])/num_shots
    v1v4 = np.sum(det_events[:,indx1] & det_events[:,indx4])/num_shots
    v2v3 = np.sum(det_events[:,indx2] & det_events[:,indx3])/num_shots
    v2v4 = np.sum(det_events[:,indx2] & det_events[:,indx4])/num_shots
    v3v4 = np.sum(det_events[:,indx3] & det_events[:,indx4])/num_shots

    v1v2v3 = np.sum(det_events[:,indx1] & det_events[:,indx2] & det_events[:,indx3] )/num_shots
    v1v2v4 = np.sum(det_events[:,indx1] & det_events[:,indx2] & det_events[:,indx4] )/num_shots
    v1v3v4 = np.sum(det_events[:,indx1] & det_events[:,indx3] & det_events[:,indx4] )/num_shots
    v2v3v4 = np.sum(det_events[:,indx2] & det_events[:,indx3] & det_events[:,indx4] )/num_shots

    v1v2v3v4 = np.sum(det_events[:,indx1] & det_events[:,indx2] & det_events[:,indx3] & det_events[:,indx4] )/num_shots

    denom  = 1 - 2 * (v1+v2) + 4 * v1v2
    denom *= 1 - 2 * (v1+v3) + 4 * v1v3
    denom *= 1 - 2 * (v1+v4) + 4 * v1v4
    denom *= 1 - 2 * (v2+v3) + 4 * v2v3 
    denom *= 1 - 2 * (v2+v4) + 4 * v2v4 
    denom *= 1 - 2 * (v3+v4) + 4 * v3v4 
    denom *= 1 - 2 * (v1+v2+v3+v4) + 4*(v1v2 + v1v3 + v1v4 + v2v3 + v2v4 + v3v4) -8*(v1v2v3 + v1v2v4 + v1v3v4 + v2v3v4) + 16*v1v2v3v4 

    numer  = (1-2*v1)*(1-2*v2)*(1-2*v3)*(1-2*v4)
    numer *= 1-2*(v1+v2+v3)+4*(v1v2+v1v3+v2v3)-8*v1v2v3
    numer *= 1-2*(v1+v2+v4)+4*(v1v2+v1v4+v2v4)-8*v1v2v4
    numer *= 1-2*(v1+v3+v4)+4*(v1v3+v1v4+v3v4)-8*v1v3v4
    numer *= 1-2*(v2+v3+v4)+4*(v2v3+v2v4+v3v4)-8*v2v3v4

    if (numer/denom)<0:
        print("Bad number in estimating pijkl for (i,j,k,l)=",(indx1,indx2,indx3,indx4))
        print("numer:",numer)


    p = 0.5-0.5*(numer/denom)**(1/8)

    if p<0:

        print("Have to set 0 for 4-pnt because p=",p)
        print("inds:",(indx1,indx2,indx3,indx4))
        
        p=0

    if np.isnan(p):
        print("Have to set 0 for 4-pnt")
        print("inds:",(indx1,indx2,indx3,indx4))
        p=0

    return p