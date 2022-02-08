# include "Kernels.cuh"

__global__ void createMiniDoubletsInGPU(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::objectRanges& rangesInGPU)
{
    int blockxSize = blockDim.x*gridDim.x;
    int blockySize = blockDim.y*gridDim.y;
    for(int lowerModuleArrayIndex = blockIdx.y * blockDim.y + threadIdx.y; lowerModuleArrayIndex< (*modulesInGPU.nLowerModules); lowerModuleArrayIndex += blockySize)
    {

        int lowerModuleIndex = modulesInGPU.lowerModuleIndices[lowerModuleArrayIndex];
        int upperModuleIndex = modulesInGPU.partnerModuleIndex(lowerModuleIndex);

        if(rangesInGPU.hitRanges[lowerModuleIndex * 2] == -1) continue; //return;
        if(rangesInGPU.hitRanges[upperModuleIndex * 2] == -1) continue; //return;
        unsigned int nLowerHits = rangesInGPU.hitRanges[lowerModuleIndex * 2 + 1] - rangesInGPU.hitRanges[lowerModuleIndex * 2] + 1;
        unsigned int nUpperHits = rangesInGPU.hitRanges[upperModuleIndex * 2 + 1] - rangesInGPU.hitRanges[upperModuleIndex * 2] + 1;
        int limit = nUpperHits*nLowerHits;
        for(int hitIndex = blockIdx.x * blockDim.x + threadIdx.x; hitIndex< limit; hitIndex += blockxSize)
        {
            int lowerHitIndex =  hitIndex / nUpperHits;
            int upperHitIndex =  hitIndex % nUpperHits;
            if(upperHitIndex >= nUpperHits) continue; //return;

            unsigned int lowerHitArrayIndex = rangesInGPU.hitRanges[lowerModuleIndex * 2] + lowerHitIndex;
            unsigned int upperHitArrayIndex = rangesInGPU.hitRanges[upperModuleIndex * 2] + upperHitIndex;

            float dz, drt, dphi, dphichange, shiftedX, shiftedY, shiftedZ, noShiftedDz, noShiftedDphi, noShiftedDphiChange;

#ifdef CUT_VALUE_DEBUG
            float dzCut, drtCut, miniCut;
            bool success = runMiniDoubletDefaultAlgo(modulesInGPU, hitsInGPU, lowerModuleIndex, lowerHitArrayIndex, upperHitArrayIndex, dz,  drt, dphi, dphichange, shiftedX, shiftedY, shiftedZ, noShiftedDz, noShiftedDphi, noShiftedDphiChange, dzCut, drtCut, miniCut);
#else
            bool success = runMiniDoubletDefaultAlgo(modulesInGPU, hitsInGPU, lowerModuleIndex, lowerHitArrayIndex, upperHitArrayIndex, dz, dphi, dphichange, shiftedX, shiftedY, shiftedZ, noShiftedDz, noShiftedDphi, noShiftedDphiChange);
#endif

            if(success)
            {
                unsigned int mdModuleIndex = atomicAdd(&mdsInGPU.nMDs[lowerModuleIndex],1);
                if(mdModuleIndex >= N_MAX_MD_PER_MODULES)
                {
                    #ifdef Warnings
                    if(mdModuleIndex == N_MAX_MD_PER_MODULES)
                        printf("Mini-doublet excess alert! Module index =  %d\n",lowerModuleIndex);
                    #endif
                }
                else
                {
                    unsigned int mdIndex = lowerModuleIndex * N_MAX_MD_PER_MODULES + mdModuleIndex;
#ifdef CUT_VALUE_DEBUG
                    addMDToMemory(mdsInGPU,hitsInGPU, modulesInGPU, lowerHitArrayIndex, upperHitArrayIndex, lowerModuleIndex, dz,drt, dphi, dphichange, shiftedX, shiftedY, shiftedZ, noShiftedDz, noShiftedDphi, noShiftedDphiChange, dzCut, drtCut, miniCut, mdIndex);
#else
                    addMDToMemory(mdsInGPU,hitsInGPU, modulesInGPU, lowerHitArrayIndex, upperHitArrayIndex, lowerModuleIndex, dz, dphi, dphichange, shiftedX, shiftedY, shiftedZ, noShiftedDz, noShiftedDphi, noShiftedDphiChange, mdIndex);
#endif

                }

            }
        }
    }
}
__global__ void createSegmentsInGPU(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::objectRanges& rangesInGPU)
{
    int blockxSize = blockDim.x*gridDim.x;
    int blockySize = blockDim.y*gridDim.y;
    int blockzSize = blockDim.z*gridDim.z;
    for(int innerLowerModuleArrayIdx = blockIdx.z * blockDim.z + threadIdx.z; innerLowerModuleArrayIdx< (*modulesInGPU.nLowerModules); innerLowerModuleArrayIdx += blockzSize){
    uint16_t innerLowerModuleIndex = modulesInGPU.lowerModuleIndices[innerLowerModuleArrayIdx];

    unsigned int nConnectedModules = modulesInGPU.nConnectedModules[innerLowerModuleIndex];

    for(int outerLowerModuleArrayIdx = blockIdx.y * blockDim.y + threadIdx.y; outerLowerModuleArrayIdx< nConnectedModules; outerLowerModuleArrayIdx += blockySize){
    //if(outerLowerModuleArrayIdx >= nConnectedModules) continue;//return;

    uint16_t outerLowerModuleIndex = modulesInGPU.moduleMap[innerLowerModuleIndex * MAX_CONNECTED_MODULES + outerLowerModuleArrayIdx];

    unsigned int nInnerMDs = mdsInGPU.nMDs[innerLowerModuleIndex] > N_MAX_MD_PER_MODULES ? N_MAX_MD_PER_MODULES : mdsInGPU.nMDs[innerLowerModuleIndex];
    unsigned int nOuterMDs = mdsInGPU.nMDs[outerLowerModuleIndex] > N_MAX_MD_PER_MODULES ? N_MAX_MD_PER_MODULES : mdsInGPU.nMDs[outerLowerModuleIndex];

    //if (nInnerMDs*nOuterMDs == 0) return;
    int limit = nInnerMDs*nOuterMDs;
    if (limit == 0) continue;
    for(int hitIndex = blockIdx.x * blockDim.x + threadIdx.x; hitIndex< limit; hitIndex += blockxSize){
    int innerMDArrayIdx = hitIndex / nOuterMDs;
    int outerMDArrayIdx = hitIndex % nOuterMDs;
    //int innerMDArrayIdx = (blockIdx.x * blockDim.x + threadIdx.x) / nOuterMDs;
    //int outerMDArrayIdx = (blockIdx.x * blockDim.x + threadIdx.x) % nOuterMDs;

    //if(innerMDArrayIdx >= nInnerMDs) continue;
    if(outerMDArrayIdx >= nOuterMDs) continue;

    unsigned int innerMDIndex = rangesInGPU.mdRanges[innerLowerModuleIndex * 2] + innerMDArrayIdx;
    unsigned int outerMDIndex = rangesInGPU.mdRanges[outerLowerModuleIndex * 2] + outerMDArrayIdx;

    float zIn, zOut, rtIn, rtOut, dPhi, dPhiMin, dPhiMax, dPhiChange, dPhiChangeMin, dPhiChangeMax, dAlphaInnerMDSegment, dAlphaOuterMDSegment, dAlphaInnerMDOuterMD;

    unsigned int innerMiniDoubletAnchorHitIndex, outerMiniDoubletAnchorHitIndex;

    dPhiMin = 0;
    dPhiMax = 0;
    dPhiChangeMin = 0;
    dPhiChangeMax = 0;
    float zLo, zHi, rtLo, rtHi, sdCut, dAlphaInnerMDSegmentThreshold, dAlphaOuterMDSegmentThreshold, dAlphaInnerMDOuterMDThreshold;

    bool success = runSegmentDefaultAlgo(modulesInGPU, hitsInGPU, mdsInGPU, innerLowerModuleIndex, outerLowerModuleIndex, innerMDIndex, outerMDIndex, zIn, zOut, rtIn, rtOut, dPhi, dPhiMin, dPhiMax, dPhiChange, dPhiChangeMin, dPhiChangeMax, dAlphaInnerMDSegment, dAlphaOuterMDSegment, dAlphaInnerMDOuterMD, zLo, zHi, rtLo, rtHi, sdCut, dAlphaInnerMDSegmentThreshold, dAlphaOuterMDSegmentThreshold,
            dAlphaInnerMDOuterMDThreshold, innerMiniDoubletAnchorHitIndex, outerMiniDoubletAnchorHitIndex);

    if(success)
    {
        unsigned int segmentModuleIdx = atomicAdd(&segmentsInGPU.nSegments[innerLowerModuleIndex],1);
        if(segmentModuleIdx >= N_MAX_SEGMENTS_PER_MODULE)
        {
            #ifdef Warnings
            if(segmentModuleIdx == N_MAX_SEGMENTS_PER_MODULE)
                printf("Segment excess alert! Module index = %d\n",innerLowerModuleIndex);
            #endif
        }
        else
        {
            unsigned int segmentIdx = innerLowerModuleIndex * N_MAX_SEGMENTS_PER_MODULE + segmentModuleIdx;
#ifdef CUT_VALUE_DEBUG
            addSegmentToMemory(segmentsInGPU,innerMDIndex, outerMDIndex,innerLowerModuleIndex, outerLowerModuleIndex, innerMiniDoubletAnchorHitIndex, outerMiniDoubletAnchorHitIndex, dPhi, dPhiMin, dPhiMax, dPhiChange, dPhiChangeMin, dPhiChangeMax, zIn, zOut, rtIn, rtOut, dAlphaInnerMDSegment, dAlphaOuterMDSegment, dAlphaInnerMDOuterMD, zLo, zHi, rtLo, rtHi, sdCut, dAlphaInnerMDSegmentThreshold, dAlphaOuterMDSegmentThreshold,
                dAlphaInnerMDOuterMDThreshold, segmentIdx);
#else
            addSegmentToMemory(segmentsInGPU,innerMDIndex, outerMDIndex,innerLowerModuleIndex, outerLowerModuleIndex, innerMiniDoubletAnchorHitIndex, outerMiniDoubletAnchorHitIndex, dPhi, dPhiMin, dPhiMax, dPhiChange, dPhiChangeMin, dPhiChangeMax, segmentIdx);
#endif

        }
    }
}
}
}
}

__global__ void createTripletsInGPU(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::triplets& tripletsInGPU, unsigned int *index_gpu,int nonZeroModules)
{
    int blockxSize = blockDim.x*gridDim.x;
    int blockySize = blockDim.y*gridDim.y;
    int blockzSize = blockDim.z*gridDim.z;
    for(int innerLowerModuleArrayIdx = blockIdx.z * blockDim.z + threadIdx.z; innerLowerModuleArrayIdx< nonZeroModules; innerLowerModuleArrayIdx += blockzSize){
  //int innerInnerLowerModuleArrayIndex = index_gpu[blockIdx.z * blockDim.z + threadIdx.z];
  int innerInnerLowerModuleArrayIndex = index_gpu[innerLowerModuleArrayIdx];
  if(innerInnerLowerModuleArrayIndex >= *modulesInGPU.nLowerModules) continue;

  unsigned int innerInnerLowerModuleIndex = modulesInGPU.lowerModuleIndices[innerInnerLowerModuleArrayIndex];
  unsigned int nConnectedModules = modulesInGPU.nConnectedModules[innerInnerLowerModuleIndex];
  if(nConnectedModules == 0) continue;

  unsigned int nInnerSegments = segmentsInGPU.nSegments[innerInnerLowerModuleIndex] > N_MAX_SEGMENTS_PER_MODULE ? N_MAX_SEGMENTS_PER_MODULE : segmentsInGPU.nSegments[innerInnerLowerModuleIndex];

    for(int innerSegmentArrayIndex = blockIdx.y * blockDim.y + threadIdx.y; innerSegmentArrayIndex< nInnerSegments; innerSegmentArrayIndex += blockySize){
  //int innerSegmentArrayIndex = blockIdx.y * blockDim.y + threadIdx.y;
  //int outerSegmentArrayIndex = blockIdx.x * blockDim.x + threadIdx.x;

 // if(innerSegmentArrayIndex >= nInnerSegments) continue;

  unsigned int innerSegmentIndex = innerInnerLowerModuleIndex * N_MAX_SEGMENTS_PER_MODULE + innerSegmentArrayIndex;

  //middle lower module - outer lower module of inner segment
  unsigned int middleLowerModuleIndex = segmentsInGPU.outerLowerModuleIndices[innerSegmentIndex];

  unsigned int nOuterSegments = segmentsInGPU.nSegments[middleLowerModuleIndex] > N_MAX_SEGMENTS_PER_MODULE ? N_MAX_SEGMENTS_PER_MODULE : segmentsInGPU.nSegments[middleLowerModuleIndex];
    for(int outerSegmentArrayIndex = blockIdx.x * blockDim.x + threadIdx.x; outerSegmentArrayIndex< nOuterSegments; outerSegmentArrayIndex += blockxSize){
  //if(outerSegmentArrayIndex >= nOuterSegments) continue;

  unsigned int outerSegmentIndex = middleLowerModuleIndex * N_MAX_SEGMENTS_PER_MODULE + outerSegmentArrayIndex;
  unsigned int outerOuterLowerModuleIndex = segmentsInGPU.outerLowerModuleIndices[outerSegmentIndex];

  float zOut,rtOut,deltaPhiPos,deltaPhi,betaIn,betaOut, pt_beta;
  float zLo, zHi, rtLo, rtHi, zLoPointed, zHiPointed, sdlCut, betaInCut, betaOutCut, deltaBetaCut, kZ;

    bool success = runTripletDefaultAlgo(modulesInGPU, hitsInGPU, mdsInGPU, segmentsInGPU, innerInnerLowerModuleIndex, middleLowerModuleIndex, outerOuterLowerModuleIndex, innerSegmentIndex, outerSegmentIndex, zOut, rtOut, deltaPhiPos, deltaPhi, betaIn, betaOut, pt_beta, zLo, zHi, rtLo, rtHi, zLoPointed, zHiPointed, sdlCut, betaInCut, betaOutCut, deltaBetaCut, kZ);

  if(success)
    {
      unsigned int tripletModuleIndex = atomicAdd(&tripletsInGPU.nTriplets[innerInnerLowerModuleArrayIndex], 1);
      if(tripletModuleIndex >= N_MAX_TRIPLETS_PER_MODULE)
      {
          #ifdef Warnings
          if(tripletModuleIndex == N_MAX_TRIPLETS_PER_MODULE)
              printf("Triplet excess alert! Module index = %d\n",innerInnerLowerModuleIndex);
          #endif
      }
      unsigned int tripletIndex = innerInnerLowerModuleArrayIndex * N_MAX_TRIPLETS_PER_MODULE + tripletModuleIndex;
#ifdef CUT_VALUE_DEBUG

        addTripletToMemory(modulesInGPU, hitsInGPU, mdsInGPU, segmentsInGPU, tripletsInGPU, innerSegmentIndex, outerSegmentIndex, innerInnerLowerModuleIndex, middleLowerModuleIndex, outerOuterLowerModuleIndex, zOut, rtOut, deltaPhiPos, deltaPhi, betaIn, betaOut,pt_beta, zLo,zHi, rtLo, rtHi, zLoPointed, zHiPointed, sdlCut, betaInCut, betaOutCut, deltaBetaCut, kZ, tripletIndex);

#else
      addTripletToMemory(modulesInGPU, hitsInGPU, mdsInGPU, segmentsInGPU, tripletsInGPU, innerSegmentIndex, outerSegmentIndex, innerInnerLowerModuleIndex, middleLowerModuleIndex, outerOuterLowerModuleIndex, betaIn, betaOut, pt_beta, tripletIndex);
#endif
    }
}
}}}
__device__ inline int checkHitspT5(unsigned int ix, unsigned int jx,struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::hits& hitsInGPU)
{
    int phits1[4] = {-1,-1,-1,-1};
    int phits2[4] = {-1,-1,-1,-1};
    phits1[0] = hitsInGPU.idxs[mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*ix]]];
    phits1[1] = hitsInGPU.idxs[mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*ix+1]]];
    phits1[2] = hitsInGPU.idxs[mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*ix]+1]];
    phits1[3] = hitsInGPU.idxs[mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*ix+1]+1]];

    phits2[0] = hitsInGPU.idxs[mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*jx]]];
    phits2[1] = hitsInGPU.idxs[mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*jx+1]]];
    phits2[2] = hitsInGPU.idxs[mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*jx]+1]];
    phits2[3] = hitsInGPU.idxs[mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*jx+1]+1]];

    int npMatched =0;

    for (int i =0; i<4;i++)
    {
        bool pmatched = false;
        if(phits1[i] == -1){continue;}
        for (int j =0; j<4; j++)
        {
            if(phits2[j] == -1){continue;}
            if(phits1[i] == phits2[j]){pmatched = true; break;}
        }
        if(pmatched){npMatched++;}
    }
    return npMatched;
}

__global__ void addpT5asTrackCandidateInGPU(struct SDL::modules& modulesInGPU, struct SDL::pixelQuintuplets& pixelQuintupletsInGPU, struct SDL::trackCandidates& trackCandidatesInGPU, struct SDL::segments& segmentsInGPU, struct SDL::triplets& tripletsInGPU,struct SDL::quintuplets& quintupletsInGPU)
{
    unsigned int pixelLowerModuleArrayIndex = *modulesInGPU.nLowerModules;
    unsigned int nPixelQuintuplets = *pixelQuintupletsInGPU.nPixelQuintuplets;
    for(int pixelQuintupletIndex = blockIdx.x * blockDim.x + threadIdx.x; pixelQuintupletIndex < nPixelQuintuplets; pixelQuintupletIndex += blockDim.x*gridDim.x)
    {
        if(pixelQuintupletsInGPU.isDup[pixelQuintupletIndex])
        {
            continue;
        }
        unsigned int trackCandidateIdx = atomicAdd(trackCandidatesInGPU.nTrackCandidates,1);
        atomicAdd(trackCandidatesInGPU.nTrackCandidatespT5,1);


#ifdef TRACK_EXTENSIONS
        float radius = 0.5f*(__H2F(pixelQuintupletsInGPU.pixelRadius[pixelQuintupletIndex]) + __H2F(pixelQuintupletsInGPU.quintupletRadius[pixelQuintupletIndex]));
        addTrackCandidateToMemory(trackCandidatesInGPU, 7/*track candidate type pT5=7*/, pixelQuintupletsInGPU.pixelIndices[pixelQuintupletIndex], pixelQuintupletsInGPU.T5Indices[pixelQuintupletIndex], &pixelQuintupletsInGPU.logicalLayers[7 * pixelQuintupletIndex], &pixelQuintupletsInGPU.lowerModuleIndices[7 * pixelQuintupletIndex], &pixelQuintupletsInGPU.hitIndices[14 * pixelQuintupletIndex], __H2F(pixelQuintupletsInGPU.centerX[pixelQuintupletIndex]),
                            __H2F(pixelQuintupletsInGPU.centerY[pixelQuintupletIndex]),radius , trackCandidateIdx);        
#else
        addTrackCandidateToMemory(trackCandidatesInGPU, 7/*track candidate type pT5=7*/, pixelQuintupletsInGPU.pixelIndices[pixelQuintupletIndex], pixelQuintupletsInGPU.T5Indices[pixelQuintupletIndex], trackCandidateIdx);        
#endif
    }
}

__global__ void addpT3asTrackCandidateInGPU(struct SDL::modules& modulesInGPU, SDL::pixelTriplets& pixelTripletsInGPU, struct SDL::trackCandidates& trackCandidatesInGPU,struct SDL::segments& segmentsInGPU, struct SDL::pixelQuintuplets& pixelQuintupletsInGPU)
{
    unsigned int pixelLowerModuleArrayIndex = *modulesInGPU.nLowerModules;
    unsigned int nPixelTriplets = *pixelTripletsInGPU.nPixelTriplets;
    for(int pixelTripletArrayIndex = blockIdx.x * blockDim.x + threadIdx.x; pixelTripletArrayIndex < nPixelTriplets; pixelTripletArrayIndex += blockDim.x*gridDim.x)
    {
        //if(pixelTripletArrayIndex >= nPixelTriplets) return;
        int pixelTripletIndex = pixelTripletArrayIndex;
        if(pixelTripletsInGPU.isDup[pixelTripletIndex])  
        {
            continue;
        }
#ifdef Crossclean_pT3
    //cross cleaning step
        float eta1 = __H2F(pixelTripletsInGPU.eta_pix[pixelTripletIndex]); 
        float phi1 = __H2F(pixelTripletsInGPU.phi_pix[pixelTripletIndex]); 
        int pixelModuleIndex = *modulesInGPU.nModules - 1;
        unsigned int prefix = pixelModuleIndex*N_MAX_SEGMENTS_PER_MODULE;
        bool end= false;
        for (unsigned int jx=0; jx<*pixelQuintupletsInGPU.nPixelQuintuplets; jx++)
        {
            unsigned int pLS_jx = pixelQuintupletsInGPU.pixelIndices[jx];
            float eta2 = segmentsInGPU.eta[pLS_jx - prefix];
            float phi2 = segmentsInGPU.phi[pLS_jx - prefix];
            float dEta = abs(eta1-eta2);
            float dPhi = abs(phi1-phi2);
            if(dPhi > float(M_PI)){dPhi = dPhi - 2*float(M_PI);}
            float dR2 = dEta*dEta + dPhi*dPhi;
            if(dR2 < 1e-5f) {end=true; break;}
        }
        if(end) continue;
#endif
    
        unsigned int trackCandidateIdx = atomicAdd(trackCandidatesInGPU.nTrackCandidates,1);
        atomicAdd(trackCandidatesInGPU.nTrackCandidatespT3,1);

#ifdef TRACK_EXTENSIONS
        float radius = 0.5f * (__H2F(pixelTripletsInGPU.pixelRadius[pixelTripletIndex]) + __H2F(pixelTripletsInGPU.tripletRadius[pixelTripletIndex]));
        addTrackCandidateToMemory(trackCandidatesInGPU, 5/*track candidate type pT3=5*/, pixelTripletIndex, pixelTripletIndex, &pixelTripletsInGPU.logicalLayers[5 * pixelTripletIndex], &pixelTripletsInGPU.lowerModuleIndices[5 * pixelTripletIndex], &pixelTripletsInGPU.hitIndices[10 * pixelTripletIndex], __H2F(pixelTripletsInGPU.centerX[pixelTripletIndex]), __H2F(pixelTripletsInGPU.centerY[pixelTripletIndex]),radius,
                trackCandidateIdx);
#else
        addTrackCandidateToMemory(trackCandidatesInGPU, 5/*track candidate type pT3=5*/, pixelTripletIndex, pixelTripletIndex, trackCandidateIdx);
#endif
    }
}

__global__ void addT5asTrackCandidateInGPU(struct SDL::modules& modulesInGPU, struct SDL::quintuplets& quintupletsInGPU,struct SDL::trackCandidates& trackCandidatesInGPU,struct SDL::pixelQuintuplets& pixelQuintupletsInGPU,struct SDL::pixelTriplets& pixelTripletsInGPU, struct SDL::objectRanges& rangesInGPU)
{
    int stepx = blockDim.x*gridDim.x;
    int stepy = blockDim.y*gridDim.y;
    for(int innerInnerInnerLowerModuleArrayIndex = blockIdx.y * blockDim.y + threadIdx.y; innerInnerInnerLowerModuleArrayIndex < *(modulesInGPU.nLowerModules); innerInnerInnerLowerModuleArrayIndex+=stepy)
    {
        if(rangesInGPU.quintupletModuleIndices[innerInnerInnerLowerModuleArrayIndex] == -1) continue;
        unsigned int nQuints = min(quintupletsInGPU.nQuintuplets[innerInnerInnerLowerModuleArrayIndex], N_MAX_QUINTUPLETS_PER_MODULE);
        for(int innerObjectArrayIndex = blockIdx.x * blockDim.x + threadIdx.x;innerObjectArrayIndex < nQuints;innerObjectArrayIndex+=stepx)
        {
            int quintupletIndex = rangesInGPU.quintupletModuleIndices[innerInnerInnerLowerModuleArrayIndex] + innerObjectArrayIndex;

    //don't add duplicate T5s or T5s that are accounted in pT5s
            if(quintupletsInGPU.isDup[quintupletIndex] or quintupletsInGPU.partOfPT5[quintupletIndex])
            {
                continue;//return;
            }
#ifdef Crossclean_T5
            unsigned int loop_bound = *pixelQuintupletsInGPU.nPixelQuintuplets;
            if (loop_bound < *pixelTripletsInGPU.nPixelTriplets) 
            { 
                loop_bound = *pixelTripletsInGPU.nPixelTriplets;
            }
            //cross cleaning step
            float eta1 = __H2F(quintupletsInGPU.eta[quintupletIndex]); 
            float phi1 = __H2F(quintupletsInGPU.phi[quintupletIndex]); 
            bool end = false;
            for (unsigned int jx=0; jx<loop_bound; jx++)
            {
                if(jx < *pixelQuintupletsInGPU.nPixelQuintuplets)
                {
                    float eta2 = __H2F(pixelQuintupletsInGPU.eta[jx]);
                    float phi2 = __H2F(pixelQuintupletsInGPU.phi[jx]);
                    float dEta = abs(eta1-eta2);
                    float dPhi = abs(phi1-phi2);
                    if(dPhi > float(M_PI)){dPhi = dPhi - 2*float(M_PI);}
                    float dR2 = dEta*dEta + dPhi*dPhi;
                    if(dR2 < 1e-3f) {end=true; break;}//return;
                }
                if(jx < *pixelTripletsInGPU.nPixelTriplets)
                {
                    float eta2 = __H2F(pixelTripletsInGPU.eta[jx]); 
                    float phi2 = __H2F(pixelTripletsInGPU.phi[jx]); 
                    float dEta = abs(eta1-eta2);
                    float dPhi = abs(phi1-phi2);
                    if(dPhi > float(M_PI)){dPhi = dPhi - 2*float(M_PI);}
                    float dR2 = dEta*dEta + dPhi*dPhi;
                    if(dR2 < 1e-3f) {end=true; break;}//return;
                }
            }
            if(end) continue;
#endif
            unsigned int trackCandidateIdx = atomicAdd(trackCandidatesInGPU.nTrackCandidates,1);
            atomicAdd(trackCandidatesInGPU.nTrackCandidatesT5,1);

#ifdef TRACK_EXTENSIONS
            addTrackCandidateToMemory(trackCandidatesInGPU, 4/*track candidate type T5=4*/, quintupletIndex, quintupletIndex, &quintupletsInGPU.logicalLayers[5 * quintupletIndex], &quintupletsInGPU.lowerModuleIndices[5 * quintupletIndex], &quintupletsInGPU.hitIndices[10 * quintupletIndex], quintupletsInGPU.regressionG[quintupletIndex], quintupletsInGPU.regressionF[quintupletIndex], quintupletsInGPU.regressionRadius[quintupletIndex], trackCandidateIdx);
#else
            addTrackCandidateToMemory(trackCandidatesInGPU, 4/*track candidate type T5=4*/, quintupletIndex, quintupletIndex, trackCandidateIdx);

#endif
        }
    }
}

__global__ void addpLSasTrackCandidateInGPU(struct SDL::modules& modulesInGPU, struct SDL::pixelTriplets& pixelTripletsInGPU, struct SDL::trackCandidates& trackCandidatesInGPU,struct SDL::segments& segmentsInGPU, struct SDL::pixelQuintuplets& pixelQuintupletsInGPU,struct SDL::miniDoublets& mdsInGPU, struct SDL::hits& hitsInGPU, struct SDL::quintuplets& quintupletsInGPU)
{
    //int pixelArrayIndex = blockIdx.x * blockDim.x + threadIdx.x;
    int step = blockDim.x*gridDim.x;
    unsigned int pixelLowerModuleArrayIndex = *modulesInGPU.nLowerModules;
    int pixelModuleIndex = *modulesInGPU.nModules - 1;
    unsigned int nPixels = segmentsInGPU.nSegments[pixelModuleIndex];
    for(int pixelArrayIndex = blockIdx.x * blockDim.x + threadIdx.x;pixelArrayIndex < nPixels;  pixelArrayIndex +=step)
    {

        if((!segmentsInGPU.isQuad[pixelArrayIndex]) || (segmentsInGPU.isDup[pixelArrayIndex]))
        {
            continue;//return;
        }
        if(segmentsInGPU.score[pixelArrayIndex] > 120){continue;}
        //cross cleaning step

        float eta1 = segmentsInGPU.eta[pixelArrayIndex];
        float phi1 = segmentsInGPU.phi[pixelArrayIndex];
        unsigned int prefix = pixelModuleIndex*N_MAX_SEGMENTS_PER_MODULE;

        unsigned int loop_bound = *pixelQuintupletsInGPU.nPixelQuintuplets;
        if (loop_bound < *pixelTripletsInGPU.nPixelTriplets) { loop_bound = *pixelTripletsInGPU.nPixelTriplets;}

        unsigned int nTrackCandidates = *(trackCandidatesInGPU.nTrackCandidates);
        bool end = false;
        for (unsigned int jx=0; jx<nTrackCandidates; jx++)
        {
            unsigned int trackCandidateIndex = jx;
            short type = trackCandidatesInGPU.trackCandidateType[trackCandidateIndex];
            unsigned int innerTrackletIdx = trackCandidatesInGPU.objectIndices[2 * trackCandidateIndex];
            if(type == 4)
            {
                unsigned int quintupletIndex = innerTrackletIdx;//trackCandidatesInGPU.objectIndices[2*jx];//T5 index
                float eta2 = __H2F(quintupletsInGPU.eta[quintupletIndex]);
                float phi2 = __H2F(quintupletsInGPU.phi[quintupletIndex]);
                float dEta = abs(eta1-eta2);
                float dPhi = abs(phi1-phi2);
                if(dPhi > float(M_PI)){dPhi = dPhi - 2*float(M_PI);}
                float dR2 = dEta*dEta + dPhi*dPhi;
                if(dR2 < 1e-3f) {end=true;break;}//return;
            }
        }
        if(end) continue;

        for (unsigned int jx=0; jx<loop_bound; jx++)
        {
            if(jx < *pixelQuintupletsInGPU.nPixelQuintuplets)
            {
                if(!pixelQuintupletsInGPU.isDup[jx])
                {
                    unsigned int pLS_jx = pixelQuintupletsInGPU.pixelIndices[jx];
                    int npMatched = checkHitspT5(prefix+pixelArrayIndex,pLS_jx,mdsInGPU,segmentsInGPU,hitsInGPU);
                    if(npMatched >0)
                    {
                        end=true;
                        break;
                    }
                    float eta2 = segmentsInGPU.eta[pLS_jx - prefix];
                    float phi2 = segmentsInGPU.phi[pLS_jx - prefix];
                    float dEta = abs(eta1-eta2);
                    float dPhi = abs(phi1-phi2);
                    if(dPhi > float(M_PI)){dPhi = dPhi - 2*float(M_PI);}
                    float dR2 = dEta*dEta + dPhi*dPhi;
                    if(dR2 < 0.000001f) {end=true; break;}//return;
                }
            }
            if(jx < *pixelTripletsInGPU.nPixelTriplets)
            {
                if(!pixelTripletsInGPU.isDup[jx])
                {
                    int pLS_jx = pixelTripletsInGPU.pixelSegmentIndices[jx];
                    int npMatched = checkHitspT5(prefix+pixelArrayIndex,pLS_jx,mdsInGPU,segmentsInGPU,hitsInGPU);
                    if(npMatched >0)
                    {
                        end=true;
                        break;
                    }
                    float eta2 = __H2F(pixelTripletsInGPU.eta_pix[jx]);
                    float phi2 = __H2F(pixelTripletsInGPU.phi_pix[jx]);
                    float dEta = abs(eta1-eta2);
                    float dPhi = abs(phi1-phi2);
                    if(dPhi > float(M_PI)){dPhi = dPhi - 2*float(M_PI);}
                    float dR2 = dEta*dEta + dPhi*dPhi;
                    if(dR2 < 0.000001f) {end=true; break;}//return;
                }
            }
        }
        if(end) continue;

        unsigned int trackCandidateIdx = atomicAdd(trackCandidatesInGPU.nTrackCandidates,1);
        atomicAdd(trackCandidatesInGPU.nTrackCandidatespLS,1);
        addTrackCandidateToMemory(trackCandidatesInGPU, 8/*track candidate type pLS=8*/, pixelArrayIndex, pixelArrayIndex, trackCandidateIdx);

    }
}

__global__ void createPixelTripletsInGPUFromMap(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::triplets& tripletsInGPU, struct SDL::pixelTriplets& pixelTripletsInGPU, unsigned int* connectedPixelSize, unsigned int* connectedPixelIndex, unsigned int nPixelSegments, unsigned int* seg_pix_gpu, unsigned int* seg_pix_gpu_offset, unsigned int totalSegs)
{
    //newgrid with map
    int blockxSize = blockDim.x*gridDim.x;
    int blockySize = blockDim.y*gridDim.y;
    //unsigned int offsetIndex = blockIdx.x * blockDim.x + threadIdx.x;
    for(int offsetIndex = blockIdx.y * blockDim.y + threadIdx.y; offsetIndex< totalSegs; offsetIndex += blockySize)
    {

        int segmentModuleIndex = seg_pix_gpu_offset[offsetIndex];
        int pixelSegmentArrayIndex = seg_pix_gpu[offsetIndex];
        if(pixelSegmentArrayIndex >= nPixelSegments) continue;//return;
        if(segmentModuleIndex >= connectedPixelSize[pixelSegmentArrayIndex]) continue;//return;

        unsigned int tripletLowerModuleIndex; //index of the module that connects to this pixel
        unsigned int tempIndex = connectedPixelIndex[pixelSegmentArrayIndex] + segmentModuleIndex; //gets module array index for segment

        //these are actual module indices
        tripletLowerModuleIndex = modulesInGPU.connectedPixels[tempIndex];
        unsigned int tripletLowerModuleArrayIndex = modulesInGPU.reverseLookupLowerModuleIndices[tripletLowerModuleIndex];
        if(tripletLowerModuleArrayIndex >= *modulesInGPU.nLowerModules) continue;//return;

        unsigned int pixelModuleIndex = *modulesInGPU.nModules - 1;
        unsigned int nOuterTriplets = min(tripletsInGPU.nTriplets[tripletLowerModuleArrayIndex], N_MAX_TRIPLETS_PER_MODULE);

        if(nOuterTriplets == 0) continue;//return;
        if(modulesInGPU.moduleType[tripletLowerModuleIndex] == SDL::TwoS) continue;//return; //Removes 2S-2S

        //fetch the triplet
        for(unsigned int outerTripletArrayIndex = blockIdx.x * blockDim.x + threadIdx.x; outerTripletArrayIndex< nOuterTriplets; outerTripletArrayIndex +=blockxSize)
        {
            unsigned int pixelSegmentIndex = pixelModuleIndex * N_MAX_SEGMENTS_PER_MODULE + pixelSegmentArrayIndex;
            unsigned int outerTripletIndex = tripletLowerModuleArrayIndex * N_MAX_TRIPLETS_PER_MODULE + outerTripletArrayIndex;
            if(modulesInGPU.moduleType[tripletsInGPU.lowerModuleIndices[3 * outerTripletIndex + 1]] == SDL::TwoS) continue;//REMOVES PS-2S

            if(segmentsInGPU.isDup[pixelSegmentArrayIndex]) continue;//return;
            if(segmentsInGPU.partOfPT5[pixelSegmentArrayIndex]) continue;//return; //don't make pT3s for those pixels that are part of pT5
            if(tripletsInGPU.partOfPT5[outerTripletIndex]) continue;//return; //don't create pT3s for T3s accounted in pT5s

            float pixelRadius, pixelRadiusError, tripletRadius, rPhiChiSquared, rzChiSquared, rPhiChiSquaredInwards, centerX, centerY;
            bool success = runPixelTripletDefaultAlgo(modulesInGPU, hitsInGPU, mdsInGPU, segmentsInGPU, tripletsInGPU, pixelSegmentIndex, outerTripletIndex, pixelRadius, pixelRadiusError, tripletRadius, centerX, centerY, rzChiSquared, rPhiChiSquared, rPhiChiSquaredInwards);

            if(success)
            {
                short layer2_adjustment;
                if(modulesInGPU.layers[tripletLowerModuleIndex] == 1)
                {
                    layer2_adjustment = 1;
                } //get upper segment to be in second layer
                else if( modulesInGPU.layers[tripletLowerModuleIndex] == 2)
                {
                    layer2_adjustment = 0;
                } // get lower segment to be in second layer
                else
                {
                    continue;
                }        
                float phi = hitsInGPU.phis[mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*tripletsInGPU.segmentIndices[2*outerTripletIndex]+layer2_adjustment]]];
                float eta = hitsInGPU.etas[mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*tripletsInGPU.segmentIndices[2*outerTripletIndex]+layer2_adjustment]]];
                float eta_pix = segmentsInGPU.eta[pixelSegmentArrayIndex];
                float phi_pix = segmentsInGPU.phi[pixelSegmentArrayIndex];
                float pt = segmentsInGPU.ptIn[pixelSegmentArrayIndex];
                float score = rPhiChiSquared+rPhiChiSquaredInwards;
                unsigned int pixelTripletIndex = atomicAdd(pixelTripletsInGPU.nPixelTriplets, 1);
                if(pixelTripletIndex >= N_MAX_PIXEL_TRIPLETS)
                {
                    #ifdef Warnings
                    if(pixelTripletIndex == N_MAX_PIXEL_TRIPLETS)
                    {
                        printf("Pixel Triplet excess alert!\n");
                    }
                    #endif
                }
                else
                {
#ifdef CUT_VALUE_DEBUG
                    addPixelTripletToMemory(modulesInGPU, hitsInGPU, mdsInGPU, segmentsInGPU, tripletsInGPU, pixelTripletsInGPU, pixelSegmentIndex, outerTripletIndex, pixelRadius, pixelRadiusError, tripletRadius, centerX, centerY, rPhiChiSquared, rPhiChiSquaredInwards, rzChiSquared, pixelTripletIndex, pt, eta, phi, eta_pix, phi_pix, score);
#else
                    addPixelTripletToMemory(modulesInGPU, hitsInGPU, mdsInGPU, segmentsInGPU, tripletsInGPU, pixelTripletsInGPU, pixelSegmentIndex, outerTripletIndex, pixelRadius,tripletRadius, centerX, centerY, pixelTripletIndex, pt,eta,phi,eta_pix,phi_pix,score);
#endif
#ifdef TRACK_EXTENSIONS
                    tripletsInGPU.partOfPT3[outerTripletIndex] = true;
#endif
                }
            }
        }
    }
}

__global__ void createQuintupletsInGPU(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::triplets& tripletsInGPU, struct SDL::quintuplets& quintupletsInGPU, unsigned int* threadIdx_gpu, unsigned int* threadIdx_gpu_offset, int nTotalTriplets, struct SDL::objectRanges& rangesInGPU)
{
    int gidy = blockIdx.y * blockDim.y + threadIdx.y;
    int np = gridDim.y * blockDim.y;
    int gidx = blockIdx.x * blockDim.x + threadIdx.x;
    int npx = gridDim.x * blockDim.x;

    for (int iter=gidy; iter < nTotalTriplets; iter+=np)
    {
        int lowerModuleArray1 = threadIdx_gpu[iter];

        //this if statement never gets executed!
        if(lowerModuleArray1  >= *modulesInGPU.nLowerModules) continue;

        unsigned int nInnerTriplets = tripletsInGPU.nTriplets[lowerModuleArray1];

        unsigned int innerTripletArrayIndex = threadIdx_gpu_offset[iter];

        if(innerTripletArrayIndex >= nInnerTriplets) continue;

        unsigned int innerTripletIndex = lowerModuleArray1 * N_MAX_TRIPLETS_PER_MODULE + innerTripletArrayIndex;
        unsigned int lowerModule1 = modulesInGPU.lowerModuleIndices[lowerModuleArray1];
        //these are actual module indices!! not lower module indices!
        unsigned int lowerModule2 = tripletsInGPU.lowerModuleIndices[3 * innerTripletIndex + 1];
        unsigned int lowerModule3 = tripletsInGPU.lowerModuleIndices[3 * innerTripletIndex + 2];
        unsigned int lowerModuleArray3 = modulesInGPU.reverseLookupLowerModuleIndices[lowerModule3];
        unsigned int nOuterTriplets = min(tripletsInGPU.nTriplets[lowerModuleArray3], N_MAX_TRIPLETS_PER_MODULE);
    //int outerTripletArrayIndex=gidx;
        for (int outerTripletArrayIndex=gidx; outerTripletArrayIndex < nOuterTriplets; outerTripletArrayIndex+=npx)
        {
            unsigned int outerTripletIndex = lowerModuleArray3 * N_MAX_TRIPLETS_PER_MODULE + outerTripletArrayIndex;
            //these are actual module indices!!
            unsigned int lowerModule4 = tripletsInGPU.lowerModuleIndices[3 * outerTripletIndex + 1];
            unsigned int lowerModule5 = tripletsInGPU.lowerModuleIndices[3 * outerTripletIndex + 2];

            float innerRadius, innerRadiusMin, innerRadiusMin2S, innerRadiusMax, innerRadiusMax2S, outerRadius, outerRadiusMin, outerRadiusMin2S, outerRadiusMax, outerRadiusMax2S, bridgeRadius, bridgeRadiusMin, bridgeRadiusMin2S, bridgeRadiusMax, bridgeRadiusMax2S, regressionG, regressionF, regressionRadius, chiSquared, nonAnchorChiSquared; //required for making distributions

            bool success = runQuintupletDefaultAlgo(modulesInGPU, hitsInGPU, mdsInGPU, segmentsInGPU, tripletsInGPU, lowerModule1, lowerModule2, lowerModule3, lowerModule4, lowerModule5, innerTripletIndex, outerTripletIndex, innerRadius, innerRadiusMin, innerRadiusMax, outerRadius, outerRadiusMin, outerRadiusMax, bridgeRadius, bridgeRadiusMin, bridgeRadiusMax, innerRadiusMin2S, innerRadiusMax2S, bridgeRadiusMin2S, bridgeRadiusMax2S, outerRadiusMin2S,
            outerRadiusMax2S, regressionG, regressionF, regressionRadius, chiSquared, nonAnchorChiSquared);

            if(success)
            {
                short layer2_adjustment;
                int layer = modulesInGPU.layers[lowerModule1];
                if(layer == 1)
                {
                    layer2_adjustment = 1;
                } //get upper segment to be in second layer
                else if(layer == 2)
                {
                    layer2_adjustment = 0;
                } // get lower segment to be in second layer
                else
                {
                    return;
                } // ignore anything else TODO: move this to start, before object is made (faster)
                unsigned int quintupletModuleIndex = atomicAdd(&quintupletsInGPU.nQuintuplets[lowerModuleArray1], 1);
                if(quintupletModuleIndex >= N_MAX_QUINTUPLETS_PER_MODULE)
                {
                    #ifdef Warnings
                    if(quintupletModuleIndex ==  N_MAX_QUINTUPLETS_PER_MODULE)
                    {
                    printf("Quintuplet excess alert! Module index = %d\n", lowerModuleArray1);
                    }
                    #endif
                }
                else
                {
                    //this if statement should never get executed!
                    if(rangesInGPU.quintupletModuleIndices[lowerModuleArray1] == -1)
                    {
                        printf("Quintuplets : no memory for module at module index = %d\n", lowerModuleArray1);
                    }
                    else
                    {
                        unsigned int quintupletIndex = rangesInGPU.quintupletModuleIndices[lowerModuleArray1] +  quintupletModuleIndex;
                        float phi = hitsInGPU.phis[mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*tripletsInGPU.segmentIndices[2*innerTripletIndex+layer2_adjustment]]]];
                        float eta = hitsInGPU.etas[mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*tripletsInGPU.segmentIndices[2*innerTripletIndex+layer2_adjustment]]]];
                        float pt = (innerRadius+outerRadius)*3.8f*1.602f/(2*100*5.39f);
                        float scores = chiSquared + nonAnchorChiSquared;
#ifdef CUT_VALUE_DEBUG
                        addQuintupletToMemory(tripletsInGPU, quintupletsInGPU, innerTripletIndex, outerTripletIndex, lowerModule1, lowerModule2, lowerModule3, lowerModule4, lowerModule5, innerRadius, innerRadiusMin, innerRadiusMax, outerRadius, outerRadiusMin, outerRadiusMax, bridgeRadius, bridgeRadiusMin, bridgeRadiusMax, innerRadiusMin2S, innerRadiusMax2S, bridgeRadiusMin2S, bridgeRadiusMax2S, outerRadiusMin2S, outerRadiusMax2S, regressionG, regressionF, regressionRadius, chiSquared, nonAnchorChiSquared,
                        pt, eta, phi, scores, layer, quintupletIndex);
#else
                        addQuintupletToMemory(tripletsInGPU, quintupletsInGPU, innerTripletIndex, outerTripletIndex, lowerModule1, lowerModule2, lowerModule3, lowerModule4, lowerModule5, innerRadius, outerRadius, regressionG, regressionF, regressionRadius, pt,eta,phi,scores,layer,quintupletIndex);
#endif
#ifdef  TRACK_EXTENSIONS
                        tripletsInGPU.partOfT5[quintupletsInGPU.tripletIndices[2 * quintupletIndex]] = true;
                        tripletsInGPU.partOfT5[quintupletsInGPU.tripletIndices[2 * quintupletIndex + 1]] = true;
#endif

                    }
                }
            }
        }
    }
}

__global__ void createPixelQuintupletsInGPUFromMap(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::triplets& tripletsInGPU, struct SDL::quintuplets& quintupletsInGPU, struct SDL::pixelQuintuplets& pixelQuintupletsInGPU, unsigned int* connectedPixelSize, unsigned int* connectedPixelIndex, unsigned int nPixelSegments, unsigned int* seg_pix_gpu, unsigned int* seg_pix_gpu_offset, unsigned int totalSegs, struct SDL::objectRanges& rangesInGPU)
{
    //unsigned int offsetIndex = blockIdx.x * blockDim.x + threadIdx.x;
    int blockxSize = blockDim.x*gridDim.x;
    int blockySize = blockDim.y*gridDim.y;
    for(int offsetIndex = blockIdx.y * blockDim.y + threadIdx.y; offsetIndex< totalSegs; offsetIndex += blockySize)
    {
        int segmentModuleIndex = seg_pix_gpu_offset[offsetIndex];
        int pixelSegmentArrayIndex = seg_pix_gpu[offsetIndex];
        if(pixelSegmentArrayIndex >= nPixelSegments) continue;//return;
        if(segmentModuleIndex >= connectedPixelSize[pixelSegmentArrayIndex]) continue;//return;

        unsigned int quintupletLowerModuleIndex; //index of the module that connects to this pixel
        unsigned int tempIndex = connectedPixelIndex[pixelSegmentArrayIndex] + segmentModuleIndex; //gets module array index for segment

    //these are actual module indices
        quintupletLowerModuleIndex = modulesInGPU.connectedPixels[tempIndex];
        unsigned int quintupletLowerModuleArrayIndex = modulesInGPU.reverseLookupLowerModuleIndices[quintupletLowerModuleIndex];
        if(quintupletLowerModuleArrayIndex >= *modulesInGPU.nLowerModules) continue;//return;

        unsigned int pixelModuleIndex = *modulesInGPU.nModules - 1;
        unsigned int nOuterQuintuplets = min(quintupletsInGPU.nQuintuplets[quintupletLowerModuleArrayIndex], N_MAX_QUINTUPLETS_PER_MODULE);

        if(nOuterQuintuplets == 0) continue;//return;

        //fetch the quintuplet
        for(unsigned int outerQuintupletArrayIndex = blockIdx.x * blockDim.x + threadIdx.x; outerQuintupletArrayIndex< nOuterQuintuplets; outerQuintupletArrayIndex +=blockxSize)
        {
            unsigned int pixelSegmentIndex = pixelModuleIndex * N_MAX_SEGMENTS_PER_MODULE + pixelSegmentArrayIndex;

            unsigned int quintupletIndex = rangesInGPU.quintupletModuleIndices[quintupletLowerModuleArrayIndex] + outerQuintupletArrayIndex;

            if(segmentsInGPU.isDup[pixelSegmentArrayIndex]) continue;//return;//skip duplicated pLS
            if(quintupletsInGPU.isDup[quintupletIndex]) continue;//return; //skip duplicated T5s

            float rzChiSquared, rPhiChiSquared, rPhiChiSquaredInwards, pixelRadius, quintupletRadius, centerX, centerY;

            bool success = runPixelQuintupletDefaultAlgo(modulesInGPU, hitsInGPU, mdsInGPU, segmentsInGPU, tripletsInGPU, quintupletsInGPU, pixelSegmentIndex, quintupletIndex, rzChiSquared, rPhiChiSquared, rPhiChiSquaredInwards, pixelRadius, quintupletRadius, centerX, centerY);

            if(success)
            {
                unsigned int pixelQuintupletIndex = atomicAdd(pixelQuintupletsInGPU.nPixelQuintuplets, 1);
                if(pixelQuintupletIndex >= N_MAX_PIXEL_QUINTUPLETS)
                {
                    #ifdef Warnings
                    if(pixelQuintupletIndex == N_MAX_PIXEL_QUINTUPLETS)
                    {
                        printf("Pixel Quintuplet excess alert!\n");
                    }
                    #endif
                }
                else
                {
                    float eta = __H2F(quintupletsInGPU.eta[quintupletIndex]);
                    float phi = __H2F(quintupletsInGPU.phi[quintupletIndex]);

#ifdef CUT_VALUE_DEBUG
                    addPixelQuintupletToMemory(modulesInGPU, hitsInGPU, mdsInGPU, segmentsInGPU, quintupletsInGPU, pixelQuintupletsInGPU, pixelSegmentIndex, quintupletIndex, pixelQuintupletIndex,rzChiSquared, rPhiChiSquared, rPhiChiSquaredInwards, rPhiChiSquared, eta, phi, pixelRadius, quintupletRadius, centerX, centerY);

#else
                    addPixelQuintupletToMemory(modulesInGPU, hitsInGPU, mdsInGPU, segmentsInGPU, quintupletsInGPU, pixelQuintupletsInGPU, pixelSegmentIndex, quintupletIndex, pixelQuintupletIndex,rPhiChiSquaredInwards+rPhiChiSquared, eta,phi, pixelRadius, quintupletRadius, centerX, centerY);
#endif
                    tripletsInGPU.partOfPT5[quintupletsInGPU.tripletIndices[2 * quintupletIndex]] = true;
                    tripletsInGPU.partOfPT5[quintupletsInGPU.tripletIndices[2 * quintupletIndex + 1]] = true;
                    segmentsInGPU.partOfPT5[pixelSegmentArrayIndex] = true;
                    quintupletsInGPU.partOfPT5[quintupletIndex] = true;
                }
            }
        }
    }
}

__device__ void scoreT5(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU,struct SDL::segments& segmentsInGPU,struct SDL::triplets& tripletsInGPU, unsigned int innerTrip, unsigned int outerTrip, int layer, float* scores)
{
    int hits1[10] = {-1,-1,-1,-1,-1,-1,-1,-1,-1,-1};
    hits1[0] = mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*tripletsInGPU.segmentIndices[2*innerTrip]]]; // inner triplet inner segment inner md inner hit
    hits1[1] = mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*tripletsInGPU.segmentIndices[2*innerTrip]]+1]; // inner triplet inner segment inner md outer hit
    hits1[2] = mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*tripletsInGPU.segmentIndices[2*innerTrip]+1]]; // inner triplet inner segment outer md inner hit
    hits1[3] = mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*tripletsInGPU.segmentIndices[2*innerTrip]+1]+1]; // inner triplet inner segment outer md outer hit
    hits1[4] = mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*tripletsInGPU.segmentIndices[2*innerTrip+1]+1]]; // inner triplet outer segment outer md inner hit
    hits1[5] = mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*tripletsInGPU.segmentIndices[2*innerTrip+1]+1]+1]; // inner triplet outer segment outer md outer hit
    hits1[6] = mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*tripletsInGPU.segmentIndices[2*outerTrip+1]]]; // outer triplet outersegment inner md inner hit
    hits1[7] = mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*tripletsInGPU.segmentIndices[2*outerTrip+1]]+1]; // outer triplet outersegment inner md outer hit
    hits1[8] = mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*tripletsInGPU.segmentIndices[2*outerTrip+1]+1]]; // outer triplet outersegment outer md inner hit
    hits1[9] = mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*tripletsInGPU.segmentIndices[2*outerTrip+1]+1]+1]; // outer triplet outersegment outer md outer hit

    unsigned int mod1 = hitsInGPU.moduleIndices[hits1[0]];
    SDL::ModuleLayerType type1 = modulesInGPU.moduleLayerType[mod1];
    unsigned int mod2 = hitsInGPU.moduleIndices[hits1[6-2*layer]];//4 for layer=1 (second hit in 3rd layer), 2 for layer=2 (second hit in third layer)
    SDL::ModuleLayerType type2 = modulesInGPU.moduleLayerType[mod1];
    float r1,r2,z1,z2;
    if(type1 == 0)
    {
        //lower hit is pixel
        r1 = hitsInGPU.rts[hits1[0]];
        z1 = hitsInGPU.zs[hits1[0]];
    }
    else
    {
        //upper hit is pixel
        r1 = hitsInGPU.rts[hits1[1]];
        z1 = hitsInGPU.zs[hits1[1]];
    }
    if(type2==0)
    {
        //lower hit is pixel
        r2 = hitsInGPU.rts[hits1[6-2*layer]];
        z2 = hitsInGPU.zs[hits1[6-2*layer]];
    }
    else
    {
        r2 = hitsInGPU.rts[hits1[7-2*layer]];
        z2 = hitsInGPU.zs[hits1[7-2*layer]];
    }
    float slope_barrel = (z2-z1)/(r2-r1);
    float slope_endcap = (r2-r1)/(z2-z1);

    //least squares
    float rsum=0, zsum=0, r2sum=0,rzsum=0;
    float rsum_e=0, zsum_e=0, r2sum_e=0,rzsum_e=0;
    for(int i =0; i < 10; i++)
    {
        rsum += hitsInGPU.rts[hits1[i]];
        zsum += hitsInGPU.zs[hits1[i]];
        r2sum += hitsInGPU.rts[hits1[i]]*hitsInGPU.rts[hits1[i]];
        rzsum += hitsInGPU.rts[hits1[i]]*hitsInGPU.zs[hits1[i]];

        rsum_e += hitsInGPU.zs[hits1[i]];
        zsum_e += hitsInGPU.rts[hits1[i]];
        r2sum_e += hitsInGPU.zs[hits1[i]]*hitsInGPU.zs[hits1[i]];
        rzsum_e += hitsInGPU.zs[hits1[i]]*hitsInGPU.rts[hits1[i]];
    }
    float slope_lsq = (10*rzsum - rsum*zsum)/(10*r2sum-rsum*rsum);
    float b = (r2sum*zsum-rsum*rzsum)/(r2sum*10-rsum*rsum);
    float slope_lsq_e = (10*rzsum_e - rsum_e*zsum_e)/(10*r2sum_e-rsum_e*rsum_e);
    float b_e = (r2sum_e*zsum_e-rsum_e*rzsum_e)/(r2sum_e*10-rsum_e*rsum_e);

    float score=0;
    float score_lsq=0;
    for( int i=0; i <10; i++)
    {
        float z = hitsInGPU.zs[hits1[i]];
        float r = hitsInGPU.rts[hits1[i]]; // cm
        float subdet = modulesInGPU.subdets[hitsInGPU.moduleIndices[hits1[i]]];
        float drdz = modulesInGPU.drdzs[hitsInGPU.moduleIndices[hits1[i]]];
        float var=0;
        float var_lsq=0;
        if(subdet == 5)
        {
            // 5== barrel
            var = slope_barrel*(r-r1) - (z-z1);
            var_lsq = slope_lsq*(r-r1) - (z-z1);
        }
        else
        {
            var = slope_endcap*(z-z1) - (r-r1);
            var_lsq = slope_lsq_e*(z-z1) - (r-r1);
        }
        float err;
        if(modulesInGPU.moduleLayerType[hitsInGPU.moduleIndices[hits1[i]]]==0)
        {
            err=0.15f*cosf(atanf(drdz));//(1.5mm)^2
        }
        else
        {
            err=5.0f*cosf(atanf(drdz));
        }//(5cm)^2
        score += (var*var) / (err*err);
        score_lsq += (var_lsq*var_lsq) / (err*err);
    }
    scores[1] = score;
    scores[3] = score_lsq;
}

__device__ int inline checkHitsT5(unsigned int ix, unsigned int jx,struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::triplets& tripletsInGPU,struct SDL::quintuplets& quintupletsInGPU)
{
    unsigned int hits1[10];// = {-1,-1,-1,-1,-1,-1,-1,-1,-1,-1};
    unsigned int hits2[10];// = {-1,-1,-1,-1,-1,-1,-1,-1,-1,-1};
    hits1[0] = mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*tripletsInGPU.segmentIndices[2*quintupletsInGPU.tripletIndices[2*ix]]]]; // inner triplet inner segment inner md inner hit
    hits1[1] = mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*tripletsInGPU.segmentIndices[2*quintupletsInGPU.tripletIndices[2*ix]]]+1]; // inner triplet inner segment inner md outer hit
    hits1[2] = mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*tripletsInGPU.segmentIndices[2*quintupletsInGPU.tripletIndices[2*ix]]+1]]; // inner triplet inner segment outer md inner hit
    hits1[3] = mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*tripletsInGPU.segmentIndices[2*quintupletsInGPU.tripletIndices[2*ix]]+1]+1]; // inner triplet inner segment outer md outer hit
    hits1[4] = mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*tripletsInGPU.segmentIndices[2*quintupletsInGPU.tripletIndices[2*ix]+1]+1]]; // inner triplet outer segment outer md inner hit
    hits1[5] = mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*tripletsInGPU.segmentIndices[2*quintupletsInGPU.tripletIndices[2*ix]+1]+1]+1]; // inner triplet outer segment outer md outer hit
    hits1[6] = mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*tripletsInGPU.segmentIndices[2*quintupletsInGPU.tripletIndices[2*ix+1]+1]]]; // outer triplet outersegment inner md inner hit
    hits1[7] = mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*tripletsInGPU.segmentIndices[2*quintupletsInGPU.tripletIndices[2*ix+1]+1]]+1]; // outer triplet outersegment inner md outer hit
    hits1[8] = mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*tripletsInGPU.segmentIndices[2*quintupletsInGPU.tripletIndices[2*ix+1]+1]+1]]; // outer triplet outersegment outer md inner hit
    hits1[9] = mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*tripletsInGPU.segmentIndices[2*quintupletsInGPU.tripletIndices[2*ix+1]+1]+1]+1]; // outer triplet outersegment outer md outer hit

    hits2[0] = mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*tripletsInGPU.segmentIndices[2*quintupletsInGPU.tripletIndices[2*jx]]]]; // inner triplet inner segment inner md inner hit
    hits2[1] = mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*tripletsInGPU.segmentIndices[2*quintupletsInGPU.tripletIndices[2*jx]]]+1]; // inner triplet inner segment inner md outer hit
    hits2[2] = mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*tripletsInGPU.segmentIndices[2*quintupletsInGPU.tripletIndices[2*jx]]+1]]; // inner triplet inner segment outer md inner hit
    hits2[3] = mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*tripletsInGPU.segmentIndices[2*quintupletsInGPU.tripletIndices[2*jx]]+1]+1]; // inner triplet inner segment outer md outer hit
    hits2[4] = mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*tripletsInGPU.segmentIndices[2*quintupletsInGPU.tripletIndices[2*jx]+1]+1]]; // inner triplet outer segment outer md inner hit
    hits2[5] = mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*tripletsInGPU.segmentIndices[2*quintupletsInGPU.tripletIndices[2*jx]+1]+1]+1]; // inner triplet outer segment outer md outer hit
    hits2[6] = mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*tripletsInGPU.segmentIndices[2*quintupletsInGPU.tripletIndices[2*jx+1]+1]]]; // outer triplet outersegment inner md inner hit
    hits2[7] = mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*tripletsInGPU.segmentIndices[2*quintupletsInGPU.tripletIndices[2*jx+1]+1]]+1]; // outer triplet outersegment inner md outer hit
    hits2[8] = mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*tripletsInGPU.segmentIndices[2*quintupletsInGPU.tripletIndices[2*jx+1]+1]+1]]; // outer triplet outersegment outer md inner hit
    hits2[9] = mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*tripletsInGPU.segmentIndices[2*quintupletsInGPU.tripletIndices[2*jx+1]+1]+1]+1]; // outer triplet outersegment outer md outer hit

    int nMatched =0;
    for (int i =0; i<10;i++)
    {
        bool matched = false;
        for (int j =0; j<10; j++)
        {
            if(hits1[i] == hits2[j])
            {
                matched = true; break;
            }
        }
        if(matched){nMatched++;}
    }
    return nMatched;
}

__device__ int duplicateCounter;
__global__ void removeDupQuintupletsInGPU(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::triplets& tripletsInGPU, struct SDL::quintuplets& quintupletsInGPU,bool secondPass,struct SDL::objectRanges& rangesInGPU)
{
    int dup_count=0;
    int nLowerModules = *modulesInGPU.nLowerModules;
    int blockxSize = blockDim.x*gridDim.x;
    int blockySize = blockDim.y*gridDim.y;
    for(unsigned int lowmod1=blockIdx.y*blockDim.y+threadIdx.y; lowmod1<nLowerModules;lowmod1+=blockySize)
    {
        int nQuintuplets_lowmod1 = quintupletsInGPU.nQuintuplets[lowmod1];
        int quintupletModuleIndices_lowmod1 = rangesInGPU.quintupletModuleIndices[lowmod1];
        for(unsigned int ix1=blockIdx.x*blockDim.x+threadIdx.x; ix1<nQuintuplets_lowmod1; ix1+=blockxSize)
        {
            unsigned int ix = quintupletModuleIndices_lowmod1 + ix1;
            if(secondPass && (quintupletsInGPU.partOfPT5[ix] || quintupletsInGPU.isDup[ix]))
            {
                continue;
            }
            float pt1  = __H2F(quintupletsInGPU.pt[ix]);
            float eta1 = __H2F(quintupletsInGPU.eta[ix]);
            float phi1 = __H2F(quintupletsInGPU.phi[ix]);
            bool isDup = false;
	    float score_rphisum1 = __H2F(quintupletsInGPU.score_rphisum[ix]);
            for(unsigned int lowmod=0; lowmod<nLowerModules;lowmod++)
            {
	        int nQuintuplets_lowmod = quintupletsInGPU.nQuintuplets[lowmod];
                int quintupletModuleIndices_lowmod = rangesInGPU.quintupletModuleIndices[lowmod];
                for(unsigned int jx1=0; jx1<nQuintuplets_lowmod; jx1++)
                {
                    unsigned int jx = quintupletModuleIndices_lowmod + jx1;
                    if(ix==jx)
                    {
                        continue;
                    }
                    if(secondPass && (quintupletsInGPU.partOfPT5[jx] || quintupletsInGPU.isDup[jx]))
                    {
                        continue;
                    }
                    float pt2  = __H2F(quintupletsInGPU.pt[jx]);
                    float eta2 = __H2F(quintupletsInGPU.eta[jx]);
                    float phi2 = __H2F(quintupletsInGPU.phi[jx]);
                    float dEta = fabsf(eta1-eta2);
                    float dPhi = fabsf(phi1-phi2);
		    float score_rphisum2 = __H2F(quintupletsInGPU.score_rphisum[jx]);
                    if (dEta > 0.1f)
                    {
                        continue;
                    }
                    if(dPhi > float(M_PI)){dPhi = dPhi - 2*float(M_PI);}
                    if (abs(dPhi) > 0.1f){continue;}
                    float dR2 = dEta*dEta + dPhi*dPhi;
                    int nMatched = checkHitsT5(ix,jx,mdsInGPU,segmentsInGPU,tripletsInGPU,quintupletsInGPU);
                    if(secondPass && (dR2 < 0.001f || nMatched >= 5))
                    {
                        if(score_rphisum1 > score_rphisum2 )
                        {
                            rmQuintupletToMemory(quintupletsInGPU,ix);
                            continue;
                        }
                        if( (score_rphisum1 == score_rphisum2) && (ix<jx))
                        {
                            rmQuintupletToMemory(quintupletsInGPU,ix);
                            continue;
                         }
                    }
                    if(nMatched >=7)
                    {
                        dup_count++;
                        if( score_rphisum1 > score_rphisum2 )
                        {
                            rmQuintupletToMemory(quintupletsInGPU,ix);continue;
                        }
                        if( (score_rphisum1 == score_rphisum2) && (ix<jx))
                        {
                            rmQuintupletToMemory(quintupletsInGPU,ix);continue;
                        }
                    }
                }
            }
        }
    }
}

__device__ float scorepT3(struct SDL::modules& modulesInGPU,struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU,struct SDL::segments& segmentsInGPU,struct SDL::triplets& tripletsInGPU, unsigned int innerPix, unsigned int outerTrip, float pt, float pz)
{
    unsigned int hits1[10];// = {-1,-1,-1,-1};
    hits1[0] = mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*innerPix]];
    hits1[1] = mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*innerPix]+1];
    hits1[2] = mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*innerPix+1]];
    hits1[3] = mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*innerPix+1]+1];
    hits1[4] = mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*tripletsInGPU.segmentIndices[2*outerTrip]]];// outer trip, inner seg, inner md, inner hit
    hits1[5] = mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*tripletsInGPU.segmentIndices[2*outerTrip]]+1];// o t, is, im oh
    hits1[6] = mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*tripletsInGPU.segmentIndices[2*outerTrip]+1]]; //ot is om ih
    hits1[7] = mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*tripletsInGPU.segmentIndices[2*outerTrip]+1]+1];
    hits1[8] = mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*tripletsInGPU.segmentIndices[2*outerTrip+1]+1]];// ot os om ih
    hits1[9] = mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*tripletsInGPU.segmentIndices[2*outerTrip+1]+1]+1];

    float r1 = hitsInGPU.rts[hits1[0]];
    float z1 = hitsInGPU.zs[hits1[0]];
    float r2 = hitsInGPU.rts[hits1[3]];
    float z2 = hitsInGPU.zs[hits1[3]];

    float slope_barrel = (z2-z1)/(r2-r1);
    float slope_endcap = (r2-r1)/(z2-z1);

    float score = 0;
    for(unsigned int i=4; i <10; i++)
    {
        float z = hitsInGPU.zs[hits1[i]];
        float r = hitsInGPU.rts[hits1[i]]; // cm
        float subdet = modulesInGPU.subdets[hitsInGPU.moduleIndices[hits1[i]]];
        float drdz = modulesInGPU.drdzs[hitsInGPU.moduleIndices[hits1[i]]];
        float var=0.f;
        if(subdet == 5)
        {// 5== barrel
            var = slope_barrel*(r-r1) - (z-z1);
        }
        else
        {
            var = slope_endcap*(z-z1) - (r-r1);
        }
        float err;
        if(modulesInGPU.moduleLayerType[hitsInGPU.moduleIndices[hits1[i]]]==0)
        {
            err=0.15f*cosf(atanf(drdz));//(1.5mm)^2
        }
        else
        {
            err=5.0f*cosf(atanf(drdz));
        }//(5cm)^2
        score += (var*var) / (err*err);
    }
    //printf("pT3 score: %f\n",score);
    return score;
}
__device__ void checkHitspT3(unsigned int ix, unsigned int jx,struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::triplets& tripletsInGPU, struct SDL::pixelTriplets& pixelTripletsInGPU,struct SDL::hits& hitsInGPU, int* matched)
{
    int phits1[4] = {-1,-1,-1,-1};
    int phits2[4] = {-1,-1,-1,-1};
    phits1[0] = hitsInGPU.idxs[mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*pixelTripletsInGPU.pixelSegmentIndices[ix]]]];
    phits1[1] = hitsInGPU.idxs[mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*pixelTripletsInGPU.pixelSegmentIndices[ix]+1]]];
    phits1[2] = hitsInGPU.idxs[mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*pixelTripletsInGPU.pixelSegmentIndices[ix]]+1]];
    phits1[3] = hitsInGPU.idxs[mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*pixelTripletsInGPU.pixelSegmentIndices[ix]+1]+1]];

    phits2[0] = hitsInGPU.idxs[mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*pixelTripletsInGPU.pixelSegmentIndices[jx]]]];
    phits2[1] = hitsInGPU.idxs[mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*pixelTripletsInGPU.pixelSegmentIndices[jx]+1]]];
    phits2[2] = hitsInGPU.idxs[mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*pixelTripletsInGPU.pixelSegmentIndices[jx]]+1]];
    phits2[3] = hitsInGPU.idxs[mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*pixelTripletsInGPU.pixelSegmentIndices[jx]+1]+1]];

    int npMatched =0;
    for (int i =0; i<4;i++)
    {
        bool pmatched = false;
        for (int j =0; j<4; j++)
        {
            if(phits1[i] == phits2[j])
            {
                pmatched = true;
                break;
            }
        }
        if(pmatched)
        {
            npMatched++;
        }
    }
    int hits1[6] = {-1,-1,-1,-1,-1,-1};
    int hits2[6] = {-1,-1,-1,-1,-1,-1};
    hits1[0] = mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*tripletsInGPU.segmentIndices[2*pixelTripletsInGPU.tripletIndices[ix]]]];// outer trip, inner seg, inner md, inner hit
    hits1[1] = mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*tripletsInGPU.segmentIndices[2*pixelTripletsInGPU.tripletIndices[ix]]]+1];// o t, is, im oh
    hits1[2] = mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*tripletsInGPU.segmentIndices[2*pixelTripletsInGPU.tripletIndices[ix]]+1]]; //ot is om ih
    hits1[3] = mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*tripletsInGPU.segmentIndices[2*pixelTripletsInGPU.tripletIndices[ix]]+1]+1];
    hits1[4] = mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*tripletsInGPU.segmentIndices[2*pixelTripletsInGPU.tripletIndices[ix]+1]+1]];// ot os om ih
    hits1[5] = mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*tripletsInGPU.segmentIndices[2*pixelTripletsInGPU.tripletIndices[ix]+1]+1]+1];

    hits2[0] = mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*tripletsInGPU.segmentIndices[2*pixelTripletsInGPU.tripletIndices[jx]]]];// outer trip, inner seg, inner md, inner hit
    hits2[1] = mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*tripletsInGPU.segmentIndices[2*pixelTripletsInGPU.tripletIndices[jx]]]+1];// o t, is, im oh
    hits2[2] = mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*tripletsInGPU.segmentIndices[2*pixelTripletsInGPU.tripletIndices[jx]]+1]]; //ot is om ih
    hits2[3] = mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*tripletsInGPU.segmentIndices[2*pixelTripletsInGPU.tripletIndices[jx]]+1]+1];
    hits2[4] = mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*tripletsInGPU.segmentIndices[2*pixelTripletsInGPU.tripletIndices[jx]+1]+1]];// ot os om ih
    hits2[5] = mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*tripletsInGPU.segmentIndices[2*pixelTripletsInGPU.tripletIndices[jx]+1]+1]+1];

    int nMatched =0;
    for (int i =0; i<6;i++)
    {
        bool matched = false;
        for (int j =0; j<6; j++)
        {
            if(hits1[i] == hits2[j])
            {
                matched = true;
                break;
            }
        }
        if(matched){nMatched++;}
    }

    matched[0] = npMatched;
    matched[1] = nMatched;
}

__device__ int duplicateCounter_pT3 =0;

__global__ void removeDupPixelTripletsInGPUFromMap(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::pixelTriplets& pixelTripletsInGPU, struct SDL::triplets& tripletsInGPU, bool secondPass)
{
    int dup_count=0;
    for (unsigned int ix=blockIdx.x*blockDim.x+threadIdx.x; ix<*pixelTripletsInGPU.nPixelTriplets; ix+=blockDim.x*gridDim.x)
    {
        bool isDup = false;
        float score1 = __H2F(pixelTripletsInGPU.score[ix]);
        for (unsigned int jx=0; jx<*pixelTripletsInGPU.nPixelTriplets; jx++)
        {
            float score2 = __H2F(pixelTripletsInGPU.score[jx]);
            if(ix==jx)
            {
                continue;
            }
            int nMatched[2];
            checkHitspT3(ix,jx,mdsInGPU,segmentsInGPU,tripletsInGPU,pixelTripletsInGPU,hitsInGPU, nMatched);
            if(((nMatched[0] + nMatched[1]) >= 5) )
            {
                dup_count++;
#ifdef TRACK_EXTENSIONS
                //check the layers
                if(pixelTripletsInGPU.logicalLayers[5*jx+2] < pixelTripletsInGPU.logicalLayers[5*ix+2])
                {
                    rmPixelTripletToMemory(pixelTripletsInGPU, ix);
                    break;
                }

                else if( pixelTripletsInGPU.logicalLayers[5*ix+2] == pixelTripletsInGPU.logicalLayers[5*jx+2] && __H2F(pixelTripletsInGPU.score[ix]) > __H2F(pixelTripletsInGPU.score[jx]))
                {
                    rmPixelTripletToMemory(pixelTripletsInGPU,ix);
                    break;
                }
                else if( pixelTripletsInGPU.logicalLayers[5*ix+2] == pixelTripletsInGPU.logicalLayers[5*jx+2] && (__H2F(pixelTripletsInGPU.score[ix]) == __H2F(pixelTripletsInGPU.score[jx])) && (ix<jx))
                {
                    rmPixelTripletToMemory(pixelTripletsInGPU,ix);
                    break;
                }
#else
                 if( score1 > score2 )
                 {
                    rmPixelTripletToMemory(pixelTripletsInGPU,ix);
                    break;                                                         
                 }
             
                 if( (score1 == score2) && (ix<jx) )
                 { 
                    rmPixelTripletToMemory(pixelTripletsInGPU,ix);
                    break;                                                                           
                }
#endif
            }
        }
    }
}

__global__ void markUsedObjects(struct SDL::modules& modulesInGPU, struct SDL::segments& segmentsInGPU, struct SDL::triplets& tripletsInGPU, struct SDL::pixelQuintuplets& pixelQuintupletsInGPU, struct SDL::quintuplets& quintupletsInGPU)
{
    for (unsigned int ix=blockIdx.x*blockDim.x+threadIdx.x; ix<*pixelQuintupletsInGPU.nPixelQuintuplets; ix+=blockDim.x*gridDim.x)
    {
        //mark the relevant T5 and pT3 here!
        if(pixelQuintupletsInGPU.isDup[ix])
        {
            continue;
        }
        unsigned int quintupletIndex = pixelQuintupletsInGPU.T5Indices[ix];
        unsigned int pixelSegmentArrayIndex = pixelQuintupletsInGPU.pixelIndices[ix]- ((*modulesInGPU.nModules - 1)* N_MAX_SEGMENTS_PER_MODULE);
    }
}

__global__ void removeDupPixelQuintupletsInGPUFromMap(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::pixelTriplets& pixelTripletsInGPU, struct SDL::triplets& tripletsInGPU, struct SDL::pixelQuintuplets& pixelQuintupletsInGPU, struct SDL::quintuplets& quintupletsInGPU, bool secondPass)
{
    //printf("running pT5 duprm\n");
    int dup_count=0;
    int nPixelQuintuplets = *pixelQuintupletsInGPU.nPixelQuintuplets;
    int blockxSize=blockDim.x*gridDim.x;
    for (unsigned int ix=blockIdx.x*blockDim.x+threadIdx.x; ix<nPixelQuintuplets; ix+=blockxSize)
    {
        bool isDup = false;
        if(secondPass && pixelQuintupletsInGPU.isDup[ix])
        {
            continue;
        }
	float score1 = __H2F(pixelQuintupletsInGPU.score[ix]);
        for (unsigned int jx=0; jx<nPixelQuintuplets; jx++)
        {
            if(ix==jx)
            {
                continue;
            }
            if(secondPass && pixelQuintupletsInGPU.isDup[jx])
            {
                continue;
            }
            unsigned int T5_ix = pixelQuintupletsInGPU.T5Indices[ix];
            unsigned int T5_jx = pixelQuintupletsInGPU.T5Indices[jx];
            unsigned int pLS_ix = pixelQuintupletsInGPU.pixelIndices[ix];
            unsigned int pLS_jx = pixelQuintupletsInGPU.pixelIndices[jx];
            int nMatched = checkHitsT5(T5_ix,T5_jx,mdsInGPU,segmentsInGPU,tripletsInGPU,quintupletsInGPU);
            int npMatched = checkHitspT5(pLS_ix,pLS_jx,mdsInGPU,segmentsInGPU,hitsInGPU);
	    float score2 = __H2F(pixelQuintupletsInGPU.score[jx]);
            if(((nMatched + npMatched) >=7))// || (secondPass && ((nMatched + npMatched) >=1)))
            {
                dup_count++;
                if( score1 > score2)
                {
                    rmPixelQuintupletToMemory(pixelQuintupletsInGPU,ix);
                    break;
                }
                if( (score1 == score2) && (ix>jx))
                {
                    rmPixelQuintupletToMemory(pixelQuintupletsInGPU,ix);
                    break;
                }
            }
        }
    }
}

__global__ void checkHitspLS(struct SDL::modules& modulesInGPU,struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::hits& hitsInGPU,bool secondpass)
{
    int counter=0;
    int pixelModuleIndex = *modulesInGPU.nModules - 1;
    unsigned int prefix = pixelModuleIndex*N_MAX_SEGMENTS_PER_MODULE;
    unsigned int nPixelSegments = segmentsInGPU.nSegments[pixelModuleIndex];
    if(nPixelSegments >  N_MAX_PIXEL_SEGMENTS_PER_MODULE)
    {
        nPixelSegments =  N_MAX_PIXEL_SEGMENTS_PER_MODULE;
    }
    for(int ix=blockIdx.x*blockDim.x+threadIdx.x;ix<nPixelSegments;ix+=blockDim.x*gridDim.x)
    {
        if(secondpass && (!segmentsInGPU.isQuad[ix] || segmentsInGPU.isDup[ix])){continue;}
        bool found=false;
        unsigned int phits1[4] ;
        //bool isQuad_ix = segmentsInGPU.isQuad[ix];
        //printf("isQuad %d\n",isQuad_ix);
        phits1[0] = hitsInGPU.idxs[mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*(prefix+ix)]]];
        phits1[1] = hitsInGPU.idxs[mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*(prefix+ix)+1]]];
        phits1[2] = hitsInGPU.idxs[mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*(prefix+ix)]+1]];
        phits1[3] = hitsInGPU.idxs[mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*(prefix+ix)+1]+1]];
        float eta_pix1 = segmentsInGPU.eta[ix];
        float phi_pix1 = segmentsInGPU.phi[ix];
        float pt1 = segmentsInGPU.ptIn[ix];
        for(int jx=0;jx<nPixelSegments;jx++)
        {
            if(secondpass && (!segmentsInGPU.isQuad[jx] || segmentsInGPU.isDup[jx])){continue;}
            if(ix==jx)
            {
                continue;
            }
            //bool isQuad_jx = segmentsInGPU.isQuad[jx];
            int quad_diff = segmentsInGPU.isQuad[ix] -segmentsInGPU.isQuad[jx];
            float ptErr_diff = segmentsInGPU.ptIn[ix] -segmentsInGPU.ptIn[jx];
            float score_diff = segmentsInGPU.score[ix] -segmentsInGPU.score[jx];
            if( (quad_diff > 0 )|| (score_diff<0 && quad_diff ==0))
            //if( (quad_diff > 0 )|| (ptErr_diff>0 && quad_diff ==0))
            {
                continue;
            }// always keep quads over trips. If they are the same, we want the object with the lower pt Error

            unsigned int phits2[4] ;
            phits2[0] = hitsInGPU.idxs[mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*(prefix+jx)]]];
            phits2[1] = hitsInGPU.idxs[mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*(prefix+jx)+1]]];
            phits2[2] = hitsInGPU.idxs[mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*(prefix+jx)]+1]];
            phits2[3] = hitsInGPU.idxs[mdsInGPU.hitIndices[2*segmentsInGPU.mdIndices[2*(prefix+jx)+1]+1]];
            float eta_pix2 = segmentsInGPU.eta[jx];
            float phi_pix2 = segmentsInGPU.phi[jx];
            float pt2 = segmentsInGPU.ptIn[jx];
            //if(abs(1/pt1 - 1/pt2)> 0.1)
            //{
            //    continue;
            //}
            int npMatched =0;
            for (int i =0; i<4;i++)
            {
                bool pmatched = false;
                for (int j =0; j<4; j++)
                {
                    if(phits1[i] == phits2[j])
                    {
                        pmatched = true;
                        break;
                    }
                }
                if(pmatched)
                {
                    npMatched++;
                }
            }
            if((npMatched ==4) && (ix < jx))
            { // if exact match, remove only 1
                found=true;
                break;
            }
            if(npMatched ==3)
            //if(npMatched >=2)
            {
                found=true;
                break;
            }
            if(secondpass){
              //printf("secondpass\n");
              if(npMatched >=1)
              {
                  found=true;
                  break;
              }
              float dEta = abs(eta_pix1-eta_pix2);
              float dPhi = abs(phi_pix1-phi_pix2);
              if(dPhi > float(M_PI)){dPhi = dPhi - 2*float(M_PI);}
              //if(abs(dPhi) > 0.03){continue;}
              //if(abs(1./pt1 - 1./pt2) > 0.5){continue;}
              float dR2 = dEta*dEta + dPhi*dPhi;
              //if(dR2 <0.0003)
              if(dR2 <0.00075f)
              {
                  found=true;
                  break;
              }
            }
        }
        if(found){counter++;rmPixelSegmentFromMemory(segmentsInGPU,ix);continue;}
    }
}

#ifdef TRACK_EXTENSIONS
#ifdef T3T3_EXTENSIONS
__global__ void createT3T3ExtendedTracksInGPU(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::triplets& tripletsInGPU, struct SDL::quintuplets& quintupletsInGPU, struct SDL::pixelTriplets& pixelTripletsInGPU, struct SDL::pixelQuintuplets& pixelQuintupletsInGPU, struct SDL::trackCandidates& trackCandidatesInGPU, struct SDL::trackExtensions& trackExtensionsInGPU, unsigned int nTrackCandidates)
{
    /*
       Changes to storage. T3-T3 extensions will be stored in contiguous (random) order!
    */
    int lowerModuleIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int firstT3ArrayIdx = blockIdx.y * blockDim.y + threadIdx.y;
    int secondT3ArrayIdx = blockIdx.z * blockDim.z + threadIdx.z;

    short constituentTCType[3];
    unsigned int constituentTCIndex[3];
    unsigned int nLayerOverlaps[2], nHitOverlaps[2];
    float rPhiChiSquared, rzChiSquared, regressionRadius, innerRadius, outerRadius;
    bool success;

    //targeting layer overlap = 2
    if(lowerModuleIdx >= *modulesInGPU.nLowerModules) return;
    if(firstT3ArrayIdx >= tripletsInGPU.nTriplets[lowerModuleIdx]) return;

    unsigned int firstT3Idx = lowerModuleIdx * N_MAX_TRIPLETS_PER_MODULE + firstT3ArrayIdx;
    if(tripletsInGPU.partOfExtension[firstT3Idx] or tripletsInGPU.partOfPT5[firstT3Idx] or tripletsInGPU.partOfT5[firstT3Idx] or tripletsInGPU.partOfPT3[firstT3Idx]) return;

    unsigned int nStaggeredModules;
    unsigned int staggeredModuleIndices[10];

    unsigned int outerLowerModuleIndex = tripletsInGPU.lowerModuleIndices[3 * firstT3Idx + 1];

    findStaggeredNeighbours(modulesInGPU, outerLowerModuleIndex, staggeredModuleIndices, nStaggeredModules);
    staggeredModuleIndices[nStaggeredModules] = outerLowerModuleIndex;
    nStaggeredModules++;

    unsigned int outerT3StartingLowerModuleIdx, secondT3Idx;

    for(size_t i = 0; i < nStaggeredModules; i++)
    {
        outerT3StartingLowerModuleIdx = modulesInGPU.reverseLookupLowerModuleIndices[staggeredModuleIndices[i]];
        if(secondT3ArrayIdx >= tripletsInGPU.nTriplets[outerT3StartingLowerModuleIdx]) continue;
   
        secondT3Idx = outerT3StartingLowerModuleIdx * N_MAX_TRIPLETS_PER_MODULE + secondT3ArrayIdx;
       if(tripletsInGPU.partOfExtension[secondT3Idx] or tripletsInGPU.partOfPT5[secondT3Idx] or tripletsInGPU.partOfT5[secondT3Idx] or tripletsInGPU.partOfPT3[secondT3Idx]) continue;

        success = runTrackExtensionDefaultAlgo(modulesInGPU, hitsInGPU, mdsInGPU, segmentsInGPU, tripletsInGPU, quintupletsInGPU, pixelTripletsInGPU, pixelQuintupletsInGPU, trackCandidatesInGPU, firstT3Idx, secondT3Idx, 3, 3, firstT3Idx, 2, constituentTCType, constituentTCIndex, nLayerOverlaps, nHitOverlaps, rPhiChiSquared, rzChiSquared, regressionRadius, innerRadius, outerRadius); 

        if(success and nLayerOverlaps[0] == 2)
        {
            unsigned int trackExtensionArrayIndex = atomicAdd(&trackExtensionsInGPU.nTrackExtensions[nTrackCandidates], 1);
            if(trackExtensionArrayIndex >= (N_MAX_T3T3_TRACK_EXTENSIONS))
            {
#ifdef Warnings
                if(trackExtensionArrayIndex == N_MAX_T3T3_TRACK_EXTENSIONS)
                {
                    printf("T3T3 track extensions overflow!\n");
                }
#endif
            }
            else
            {
                unsigned int trackExtensionIndex = nTrackCandidates * N_MAX_TRACK_EXTENSIONS_PER_TC + trackExtensionArrayIndex;
#ifdef CUT_VALUE_DEBUG
                addTrackExtensionToMemory(trackExtensionsInGPU, constituentTCType, constituentTCIndex, nLayerOverlaps, nHitOverlaps, rPhiChiSquared, rzChiSquared, regressionRadius,  innerRadius, outerRadius, trackExtensionIndex);
#else
                addTrackExtensionToMemory(trackExtensionsInGPU, constituentTCType, constituentTCIndex, nLayerOverlaps, nHitOverlaps, rPhiChiSquared, rzChiSquared, regressionRadius, trackExtensionIndex);

#endif
                trackExtensionsInGPU.isDup[trackExtensionIndex] = false;
                tripletsInGPU.partOfExtension[firstT3Idx] = true;
                tripletsInGPU.partOfExtension[secondT3Idx] = true;
            }
        }
    }

    outerLowerModuleIndex = tripletsInGPU.lowerModuleIndices[3 * firstT3Idx + 2];
    nStaggeredModules = 0;
    findStaggeredNeighbours(modulesInGPU, outerLowerModuleIndex, staggeredModuleIndices, nStaggeredModules);
    
    for(size_t i = 0; i < nStaggeredModules; i++)
    {
        outerT3StartingLowerModuleIdx = modulesInGPU.reverseLookupLowerModuleIndices[staggeredModuleIndices[i]];
        if(secondT3ArrayIdx >= tripletsInGPU.nTriplets[outerT3StartingLowerModuleIdx]) continue;
   
        secondT3Idx = outerT3StartingLowerModuleIdx * N_MAX_TRIPLETS_PER_MODULE + secondT3ArrayIdx;
        if(tripletsInGPU.partOfExtension[secondT3Idx] or tripletsInGPU.partOfPT5[secondT3Idx] or tripletsInGPU.partOfT5[secondT3Idx] or tripletsInGPU.partOfPT3[secondT3Idx]) continue;

        success = runTrackExtensionDefaultAlgo(modulesInGPU, hitsInGPU, mdsInGPU, segmentsInGPU, tripletsInGPU, quintupletsInGPU, pixelTripletsInGPU, pixelQuintupletsInGPU, trackCandidatesInGPU, firstT3Idx, secondT3Idx, 3, 3, firstT3Idx, 1, constituentTCType, constituentTCIndex, nLayerOverlaps, nHitOverlaps, rPhiChiSquared, rzChiSquared, regressionRadius, innerRadius, outerRadius); 

        if(success and nLayerOverlaps[0] == 1 and nHitOverlaps[0] != 2)        
        {
            unsigned int trackExtensionArrayIndex = atomicAdd(&trackExtensionsInGPU.nTrackExtensions[nTrackCandidates], 1);
            if(trackExtensionArrayIndex >= (N_MAX_T3T3_TRACK_EXTENSIONS))
            {
#ifdef Warnings
                if(trackExtensionArrayIndex == N_MAX_T3T3_TRACK_EXTENSIONS)
                {
                    printf("T3T3 track extensions overflow!\n");
                }
#endif
            }
            else
            {
                unsigned int trackExtensionIndex = nTrackCandidates * N_MAX_TRACK_EXTENSIONS_PER_TC + trackExtensionArrayIndex;
#ifdef CUT_VALUE_DEBUG
                addTrackExtensionToMemory(trackExtensionsInGPU, constituentTCType, constituentTCIndex, nLayerOverlaps, nHitOverlaps, rPhiChiSquared, rzChiSquared, regressionRadius,  innerRadius, outerRadius, trackExtensionIndex);
#else
                addTrackExtensionToMemory(trackExtensionsInGPU, constituentTCType, constituentTCIndex, nLayerOverlaps, nHitOverlaps, rPhiChiSquared, rzChiSquared, regressionRadius, trackExtensionIndex);
#endif
                trackExtensionsInGPU.isDup[trackExtensionIndex] = false;
                tripletsInGPU.partOfExtension[firstT3Idx] = true;
                tripletsInGPU.partOfExtension[secondT3Idx] = true;

            }
        }
    }
}
#endif

__global__ void createExtendedTracksInGPU(struct SDL::modules& modulesInGPU, struct SDL::hits& hitsInGPU, struct SDL::miniDoublets& mdsInGPU, struct SDL::segments& segmentsInGPU, struct SDL::triplets& tripletsInGPU, struct SDL::pixelTriplets& pixelTripletsInGPU, struct SDL::quintuplets& quintupletsInGPU, struct SDL::pixelQuintuplets& pixelQuintupletsInGPU, struct SDL::trackCandidates& trackCandidatesInGPU, struct SDL::trackExtensions& trackExtensionsInGPU)
{
    int tcIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int t3ArrayIdx = blockIdx.y * blockDim.y + threadIdx.y;
    int layerOverlap = blockIdx.z * blockDim.z + threadIdx.z;
    //layerOverlap
    if(layerOverlap == 0 or layerOverlap >= 3) return;
    if(tcIdx >= *(trackCandidatesInGPU.nTrackCandidates)) return;
    short tcType = trackCandidatesInGPU.trackCandidateType[tcIdx];                                
    unsigned int outerT3StartingModuleIndex;
    unsigned int outerT3Index;
    if(tcType == 8) return;
    //FIXME: Need to use staggering modules for the first outer T3 module itself!
    else if(tcType == 7 or tcType == 4)
    {
        unsigned int outerT5Index = trackCandidatesInGPU.objectIndices[2 * tcIdx + 1];
        outerT3Index = quintupletsInGPU.tripletIndices[2 * outerT5Index];
        outerT3StartingModuleIndex = quintupletsInGPU.lowerModuleIndices[5 * outerT5Index + 5 - layerOverlap];
    }
    else if(tcType == 5) //pT3
    {
        unsigned int pT3Index = trackCandidatesInGPU.objectIndices[2 * tcIdx];
        outerT3Index = pixelTripletsInGPU.tripletIndices[pT3Index];
        outerT3StartingModuleIndex = tripletsInGPU.lowerModuleIndices[3 * outerT3Index + 3 - layerOverlap];  
    }


    unsigned int outerT3StartingLowerModuleIndex = modulesInGPU.reverseLookupLowerModuleIndices[outerT3StartingModuleIndex];
    if(t3ArrayIdx >= tripletsInGPU.nTriplets[outerT3StartingLowerModuleIndex]) return;
    unsigned int t3Idx =  outerT3StartingLowerModuleIndex * N_MAX_TRIPLETS_PER_MODULE + t3ArrayIdx;
    short constituentTCType[3];
    unsigned int constituentTCIndex[3];
    unsigned int nLayerOverlaps[2], nHitOverlaps[2];
    float rzChiSquared, rPhiChiSquared, regressionRadius, innerRadius, outerRadius;

    bool success = runTrackExtensionDefaultAlgo(modulesInGPU, hitsInGPU, mdsInGPU, segmentsInGPU, tripletsInGPU, quintupletsInGPU, pixelTripletsInGPU, pixelQuintupletsInGPU, trackCandidatesInGPU, tcIdx, t3Idx, tcType, 3, outerT3Index, layerOverlap, constituentTCType, constituentTCIndex, nLayerOverlaps, nHitOverlaps, rPhiChiSquared, rzChiSquared, regressionRadius, innerRadius, outerRadius);
    if(success)
    {
        unsigned int trackExtensionArrayIndex = atomicAdd(&trackExtensionsInGPU.nTrackExtensions[tcIdx], 1);
        if(trackExtensionArrayIndex >= N_MAX_TRACK_EXTENSIONS_PER_TC)
        {
#ifdef Warnings
            if(trackExtensionArrayIndex == N_MAX_TRACK_EXTENSIONS_PER_TC)
            {
                printf("Track extensions overflow for TC index = %d\n", tcIdx);
            }
#endif
        }
        else
        {
            unsigned int trackExtensionIndex = tcIdx * N_MAX_TRACK_EXTENSIONS_PER_TC + trackExtensionArrayIndex; 
#ifdef CUT_VALUE_DEBUG
            addTrackExtensionToMemory(trackExtensionsInGPU, constituentTCType, constituentTCIndex, nLayerOverlaps, nHitOverlaps, rPhiChiSquared, rzChiSquared, regressionRadius, innerRadius, outerRadius, trackExtensionIndex);
#else
            addTrackExtensionToMemory(trackExtensionsInGPU, constituentTCType, constituentTCIndex, nLayerOverlaps, nHitOverlaps, rPhiChiSquared, rzChiSquared, regressionRadius, trackExtensionIndex);
#endif
            trackCandidatesInGPU.partOfExtension[tcIdx] = true;
            tripletsInGPU.partOfExtension[t3Idx] = true;
        }
    }
}

__global__ void cleanDuplicateExtendedTracks(struct SDL::trackExtensions& trackExtensionsInGPU, unsigned int nTrackCandidates)
{
    int trackCandidateIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if(trackCandidateIndex >= nTrackCandidates) return;
    float minRPhiChiSquared = 999999999;
    float minRZChiSquared = 0; //rz chi squared corresponding to the minimum r-phi chi squared
    unsigned int minIndex = 0;
    for(size_t i = 0; i < min(trackExtensionsInGPU.nTrackExtensions[trackCandidateIndex], N_MAX_TRACK_EXTENSIONS_PER_TC); i++)
    {
        float candidateRPhiChiSquared = __H2F(trackExtensionsInGPU.rPhiChiSquared[trackCandidateIndex * N_MAX_TRACK_EXTENSIONS_PER_TC + i]);
        float candidateRZChiSquared = __H2F(trackExtensionsInGPU.rzChiSquared[trackCandidateIndex * N_MAX_TRACK_EXTENSIONS_PER_TC + i]);

        if( candidateRPhiChiSquared < minRPhiChiSquared)
        {
            minIndex = i;
            minRPhiChiSquared = candidateRPhiChiSquared;
            minRZChiSquared = candidateRZChiSquared;
        }
        else if(candidateRPhiChiSquared == minRPhiChiSquared and candidateRZChiSquared > 0 and  candidateRZChiSquared < minRZChiSquared )
        {
            minIndex = i;
            minRPhiChiSquared = candidateRPhiChiSquared;
            minRZChiSquared = candidateRZChiSquared;
        }
    }
    trackExtensionsInGPU.isDup[N_MAX_TRACK_EXTENSIONS_PER_TC * trackCandidateIndex + minIndex] = false;
}
#endif
