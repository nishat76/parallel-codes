// NEMO5, The Nonelectrical simulation package.
// Copyright (C) 2010 Purdue University
// Authors (in alphabetical order):
//
// This package is a free software.
// It is distributed under the NEMO5 Non-Commercial License (NNCL).
// The license text is found in the subfolder 'license' in the top folder.
// To request an official license document please write to the following address:
// Purdue Research Foundation, 1281 Win Hentschel Blvd., West Lafayette, IN 47906, USA
// $Id$

//! Class: Compute potentials due to wavefunction pair products using Brute Force Method
//! Full Configuration Interaction
//! Author: Archana Tankasala atankas@purdue.edu

#ifndef POTENTIALBFM
#define POTENTIALBFM

#include "Utility.h"

using namespace Utilities;
extern Parallelize P;
extern Utility U;

//! Brute Force Method potentials class
class PotentialBFM {
 public:
  //! Potentials due to pair products to save
  dcmplx3d* pPairProductPotentials;
  //! Potentials due to pair products computed without saving
  std::vector<dcmplx> *pPairProductPotentialsVnq;
  std::vector<dcmplx> *pPairProductPotentialsVnp;
  std::vector<dcmplx> *pPairProductPotentialsVpq;

  //! Compute all potentials only once and save
  void computePotentials(dcmplx3d &rPairProductsMatrix, double* atomCoordinates,
			 uLongInt nAtoms, uShortInt nEffectiveWavefunctions,
			 uShortInt wavefunctionStep, double epsilon, double ccRadius) {
    U.separator("computePotentials");
    U.log(0)<<"\nComputing all potentials to save in memory\n";
    U.activeMemory();
    double startSavePotentials = MPI_Wtime();
    centralCellR = ccRadius;
    U.log(9)<<"Using central-cell correction dr = "<<centralCellR<<"\n";
    qr0 = r0au*q;
    dielectricConstant = epsilon;
    pPairProductPotentials = new dcmplx3d();
    dcmplx3d &rPairProductPotentials = *pPairProductPotentials;
    allocateMemory(rPairProductPotentials, 
		   P.nAtomsPerProc, nEffectiveWavefunctions);
    int numPotentials = 0;
    for(int wfi=0;wfi<nEffectiveWavefunctions;wfi+=1){
      for(int wfj=0;wfj<wfi+1;wfj+=1){
	U.log(0)<<"\tPotential for ("<<wfi*wavefunctionStep+1
		<<","<<wfj*wavefunctionStep+1<<")";
	computePotential(rPairProductsMatrix[wfi][wfj], atomCoordinates,
			 rPairProductPotentials[wfi][wfj], 
			 nAtoms, P.nAtomsPerProc, P.startAtom, P.endAtom);
	numPotentials++;
      }
    }
    U.log(9)<<"Computed potentials due to "<<numPotentials<<"\n";
    copyConjugates(rPairProductPotentials, P.startAtom, 
		   P.endAtom, nEffectiveWavefunctions);
    double endSavePotentials = MPI_Wtime();
    U.log(0)<<"Finished computing potentials for "
	    <<(nEffectiveWavefunctions*nEffectiveWavefunctions)
	    <<" wavefunction pairproducts in "
	    <<endSavePotentials-startSavePotentials<<"s.\n";
    U.activeMemory();
    U.separator("computePotentials");
  }

  //! Allocate memory for only 3 potential vectors
  //! If potentials of all wavefunction pair products are not to be saved 
  void allocateDynamicPotentialVectors() {
    pPairProductPotentialsVnq = new std::vector<dcmplx> ();
    pPairProductPotentialsVnp = new std::vector<dcmplx> ();
    pPairProductPotentialsVpq = new std::vector<dcmplx> ();
    pPairProductPotentialsVnq->resize(P.nAtomsPerProc);
    pPairProductPotentialsVnp->resize(P.nAtomsPerProc);
    pPairProductPotentialsVpq->resize(P.nAtomsPerProc);
  }

  //! Compute potentials due to wavefunction pair products 
  void computePotential(std::vector<dcmplx> &rPairProduct, double* atomCoordinates,
			std::vector<dcmplx> &rPairProductPotentials,
			uLongInt nAtoms, uLongInt nAtomsPerProc, 
			uLongInt startAtom, uLongInt endAtom) {
    double potentialStart = MPI_Wtime();
    double dr, dx, dy, dz;
    double dielectricCorrection = dielectricConstant;
    for (uLongInt i=startAtom; i<endAtom; i++) {
      rPairProductPotentials[i-startAtom] = 0;
      for (uLongInt j=0; j<nAtoms; j++) {
	dx = atomCoordinates[3*i+0] - atomCoordinates[3*j+0];
	dy = atomCoordinates[3*i+1] - atomCoordinates[3*j+1];
	dz = atomCoordinates[3*i+2] - atomCoordinates[3*j+2];
	dr = std::sqrt(dx*dx + dy*dy + dz*dz);
	dielectricCorrection=getDielectric(dr);
	if(i==j)
	  dr=centralCellR;
	  rPairProductPotentials[i-startAtom] 
	    += rPairProduct[j]/(dielectricCorrection*dr);// probably here we need to put the "-" term
      }
    }
    double potentialEnd = MPI_Wtime();
    U.log(0)<<"\n\tCompleted in "<<potentialEnd-potentialStart<<"s.\n";
  }

 private:
  static const double r0au = 4.28;
  static const double q = 1.10;
  static const double a0nm = 0.0529177208;
  double qr0;
  double dielectricConstant;
  double centralCellR;

  //! Allocate memory for 3-dimensional potential matriz
  void allocateMemory(dcmplx3d &rPairProductPotentials, 
		      uLongInt nAtomsPerProc, double nEffectiveWavefunctions) {
    rPairProductPotentials.resize(nEffectiveWavefunctions);
    for (int i=0; i<nEffectiveWavefunctions; i++) {
      rPairProductPotentials[i].resize(nEffectiveWavefunctions);
      for (int j=0; j<nEffectiveWavefunctions; j++)
        rPairProductPotentials[i][j].resize(nAtomsPerProc);
    }
    U.log(9)<<"Allocated memory for saving potentials due to all pair products\n";
    U.activeMemory();
  }

  //! Copy conjugates of potentials
  void copyConjugates(dcmplx3d &rPairProductPotentials, uLongInt startAtom, 
		      uLongInt endAtom, double nEffectiveWavefunctions) {
    U.log(9)<<"Copying conjugates of potentials for remaining pair products\n";
    for(int wfi=0;wfi<nEffectiveWavefunctions;wfi+=1){
      for(int wfj=wfi+1;wfj<nEffectiveWavefunctions;wfj+=1){
        for(uLongInt i=startAtom;i<endAtom;i++)
          rPairProductPotentials[wfi][wfj][i-startAtom] 
	    = std::conj(rPairProductPotentials[wfj][wfi][i-startAtom]); printf("checking: is the code coming here  wfi is %d and wfj is %d\n",wfi,wfj);
      }
    }
  }

  //! Get distance dependent dielectric constant
  double getDielectric(double dr) {
    double dielectricCorrection = dielectricConstant;
    double rAtomic = 0.0;
    double argument = 0.0;
    if (dr <= r0au*a0nm) {
      rAtomic=dr/a0nm;
      argument=qr0-q*rAtomic;
      dielectricCorrection=dielectricConstant*qr0/((q*rAtomic) + sinh(argument));
    }
    return dielectricCorrection;
  }

};

#endif
