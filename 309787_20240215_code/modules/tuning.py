# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 20:20:15 2023

@author: hp
"""

from discriminative import lr_classifier 
from discriminative import qlr_classifier 
from discriminative import quadratic_lr_train

from costs import compute_min_DCF 
from plots import plotDCF_lambda
from plots import plotDCF_gamma
from plots import plotDCF_c
from plots import plotDCF_C
from plots import plotDCF_gmm_comp
from plots import plotDCF_lambda_eval  
from plots import plotDCF_C_eval  
from svm import kernel_RBF_H
from svm import minimize_mod_dual 
from svm import scores_error_kernel_RBF
from svm import kernel_poly_H
from svm import scores_error_kernel_poly
from svm import compute_mod_H 
from svm import primal_model
from svm import SVM_kernel_quadratic_training 
from gmm import logpdf_GMM
from gmm import ML_GMM_LBG
import utils
from utils import apply_PCA 
import numpy as np
import scipy

# ---- ___ ----
def lambda_estimation(D,L):
    Ds =(np.array_split(D, 5, axis=1))
    Ls = (np.array_split(L, 5))
    effective_prior=(0.5*1)/(0.5*1+0.5*10)
    lambdas=np.logspace(-5, 2, num=10)
    pca_flags=[False, True]
    norm_flags=[False, True]
    llrs = []
    minDcf=[]
    dcfs=[]
    
    for pca_flag in pca_flags:
        for norm_flag in norm_flags:
            for l in lambdas:
                print("lambda:",l)
                for j in range(5):
                    print("K:",j)
                    Dtr,Ltr = np.hstack(
                        Ds[:j]+ Ds[j+1:]), np.hstack(Ls[:j] + Ls[j+1:])
        
                    Dts,Lts = np.asarray(Ds[j]) , np.asarray(Ls[j])
                    if norm_flag:
                        Dtr_new, Dts_new = utils.compute_znorm(Dtr, Dts)
                    else: 
                        Dtr_new=Dtr
                        Dts_new=Dts
                          
                    if pca_flag:
                        P= apply_PCA(Dtr_new,8)
                        Dtr_new=np.dot(P.T,Dtr_new)
                        Dts_new=np.dot(P.T,Dts_new)
                    else: 
                        Dtr_new=Dtr_new
                        Dts_new=Dts_new
                        
                    _,llr=lr_classifier(Dtr_new, Ltr, Dts_new, Lts,l,effective_prior)
                    llrs.append(llr)
        
                minDcf.append(compute_min_DCF(0.5, 1, 10, np.hstack(llrs), L))
                llrs=[]
                print(minDcf)
    plotDCF_lambda(lambdas,minDcf,"lambda","LR")


# ---- ___ ----
def lambda_estimation_quadratic(D,L):
    Ds =(np.array_split(D, 5, axis=1))
    Ls = (np.array_split(L, 5))
    effective_prior=(0.5*1)/(0.5*1+0.5*10)
    lambdas=np.logspace(-5, 2, num=10)
    pca_flags=[False, True]
    norm_flags=[False, True]
    llrs = []
    minDcf=[]
    dcfs=[]
    
    for pca_flag in pca_flags:
        for norm_flag in norm_flags:
            for l in lambdas:
                for j in range(5):
                    Dtr,Ltr = np.hstack(Ds[:j]+ Ds[j+1:]), np.hstack(Ls[:j] + Ls[j+1:])
                    Dts,Lts = np.asarray(Ds[j]) , np.asarray(Ls[j])
                    if norm_flag:
                        Dtr_new, Dts_new = utils.compute_znorm(Dtr, Dts)
                    else: 
                        Dtr_new=Dtr
                        Dts_new=Dts
                    if pca_flag:
                        P= apply_PCA(Dtr_new,8)
                        Dtr_new=np.dot(P.T,Dtr_new)
                        Dts_new=np.dot(P.T,Dts_new)
                    else: 
                        Dtr_new=Dtr_new
                        Dts_new=Dts_new

                    _,llr=qlr_classifier(Dtr_new, Ltr, Dts_new, Lts,l,effective_prior)
                    llrs.append(llr)
                minDcf.append(compute_min_DCF(0.5, 1, 10, np.hstack(llrs), L))
                llrs=[]
                print(minDcf)
    plotDCF_lambda(lambdas,minDcf,"lambda", "Quadratic_evaluation")


# ---- ___ ----
def lambda_estimation_quadratic_eval(Dtr,Ltr, Dte, Lte):
    effective_prior=(0.5*1)/(0.5*1+0.5*10)
    lambdas=np.logspace(-5, 2, num=10)
    pca_flags=[False]
    norm_flags=[True]
    llrs = []
    minDcf=[]
    minDcf_train=[]
    dcfs=[]

    for pca_flag in pca_flags:
        for norm_flag in norm_flags:
            for l in lambdas:
                print("lambda:",l)
                if norm_flag:
                    Dtr_new, Dts_new = utils.compute_znorm(Dtr, Dte)
                else:
                    Dtr_new=Dtr
                    Dts_new=Dte
                if pca_flag:
                    P= apply_PCA(Dtr_new,8)
                    Dtr_new=np.dot(P.T,Dtr_new)
                    Dts_new=np.dot(P.T,Dts_new)
                else:
                    Dtr_new=Dtr_new
                    Dts_new=Dts_new

                _,llr=qlr_classifier(Dtr_new, Ltr, Dts_new, Lte,l,effective_prior)
                llrs.append(llr)
                llrs_train = quadratic_lr_train(Dtr_new, Ltr, 5, False, True, l, effective_prior)
                minDcf.append(compute_min_DCF(0.5, 1, 10, np.hstack(llr), Lte))
                minDcf_train.append(compute_min_DCF(0.5, 1, 10, np.hstack(llrs_train), Ltr))
            llrs=[]
    plotDCF_lambda_eval(lambdas,minDcf, minDcf_train,"lambda", "Quadratic evaluation")


# ---- ___ ----
def C_estimation_linear(D,L,title):
    Ds =(np.array_split(D, 5, axis=1))
    Ls = (np.array_split(L, 5))
    Cs=np.logspace(-5, 2, num=3)
    Cs=[0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
    K = 10
    svm_scores=[]
    pca_flags=[False, True]
    norm_flags=[False, True]
    llrs = []
    minDcf=[]
    dcfs=[]
    
    for pca_flag in pca_flags:
        for norm_flag in norm_flags:
            for C in Cs:
                for j in range(5):
                    print("K:",j)
                    Dtr,Ltr = np.hstack(Ds[:j]+ Ds[j+1:]), np.hstack(Ls[:j] + Ls[j+1:])
                    Dts,Lts = np.asarray(Ds[j]) , np.asarray(Ls[j])
                    if norm_flag:
                        Dtr_new, Dts_new = utils.compute_znorm(Dtr, Dts)
                    else: 
                        Dtr_new=Dtr
                        Dts_new=Dts
                    if pca_flag:
                        P= apply_PCA(Dtr_new,8)
                        Dtr_new=np.dot(P.T,Dtr_new)
                        Dts_new=np.dot(P.T,Dts_new)
                    else: 
                        Dtr_new=Dtr_new
                        Dts_new=Dts_new
                    D,z,H=compute_mod_H(Dtr_new, Ltr, K)
                    alpha_opt,opt_dual_value= minimize_mod_dual(D,C,z,H)
                    opt_primal_value,svm_wp,sv_score=primal_model(D,alpha_opt,z,K,C, Dts_new, Lts)
                    svm_scores.append(sv_score)

                minDcf.append(compute_min_DCF(0.5, 1, 10, np.hstack(svm_scores), L))
                svm_scores=[]

    plotDCF_C(Cs,minDcf,"C", title)    
    

# ---- ___ ----
def c_gamma_estimation_rbf(D,L,pca_flag,norm_flag, title):
    Ds =(np.array_split(D, 5, axis=1))
    Ls = (np.array_split(L, 5))
    gammas = [0.1, 0.01, 0.001]
    Cs=[0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
    K = 1
    minDcf=[]
    svm_scores=[]
    llrs = []
    minDcf=[]
    dcfs=[]
    
    for i in range(len(gammas)):
        for C in Cs:
            for j in range(5):
                Dtr,Ltr = np.hstack(Ds[:j]+ Ds[j+1:]), np.hstack(Ls[:j] + Ls[j+1:])
                Dts,Lts = np.asarray(Ds[j]) , np.asarray(Ls[j])
                if norm_flag:
                    Dtr_new, Dts_new = utils.compute_znorm(Dtr, Dts)
                else: 
                    Dtr_new=Dtr
                    Dts_new=Dts
                if pca_flag:
                    P= apply_PCA(Dtr_new,8)
                    Dtr_new=np.dot(P.T,Dtr_new)
                    Dts_new=np.dot(P.T,Dts_new)
                else:
                    Dtr_new=Dtr_new
                    Dts_new=Dts_new
                z,H=kernel_RBF_H(Dtr_new, Ltr,K,gammas[i])
                alpha_opt,opt_dual_value = minimize_mod_dual(Dtr_new,C,z,H)
                sv_score,svm_wp = scores_error_kernel_RBF(alpha_opt,z,Dtr_new,Dts_new,Lts,K,gammas[i])
                svm_scores.append(sv_score)
            minDcf.append(compute_min_DCF(0.5, 1, 10, np.hstack(svm_scores), L))
            svm_scores=[]
    plotDCF_gamma(Cs,minDcf,"C", title)


# ---- ___ ----
def c_estimation_poly(D,L,c, title):
    Ds =(np.array_split(D, 5, axis=1))
    Ls = (np.array_split(L, 5))
    Cs=[0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
    K = 1
    minDcf=[]
    svm_scores=[]
    pca_flags=[False, True]
    norm_flags=[False, True]

    for pca_flag in pca_flags:
        for norm_flag in norm_flags:
            for C in Cs:
                for j in range(5):
                    Dtr,Ltr = np.hstack(Ds[:j]+ Ds[j+1:]), np.hstack(Ls[:j] + Ls[j+1:])
                    Dts,Lts = np.asarray(Ds[j]) , np.asarray(Ls[j])
                    if norm_flag:
                        Dtr_new, Dts_new = utils.compute_znorm(Dtr, Dts)
                    else: 
                        Dtr_new=Dtr
                        Dts_new=Dts
                    if pca_flag:
                        P= apply_PCA(Dtr_new,8)
                        Dtr_new=np.dot(P.T,Dtr_new)
                        Dts_new=np.dot(P.T,Dts_new)
                    else:
                        Dtr_new=Dtr_new
                        Dts_new=Dts_new
                    z,H = kernel_poly_H(Dtr_new, Ltr,c,2,K)
                
                    alpha_opt,opt_dual_value= minimize_mod_dual(Dtr_new,C,z,H)
                    sv_score,svm_wp=scores_error_kernel_poly(alpha_opt,z,Dtr_new,Dts_new,Lts,c,2,K)
                    svm_scores.append(sv_score)
                minDcf.append(compute_min_DCF(0.5, 1, 10, np.hstack(svm_scores), L))
                svm_scores=[]
    plotDCF_C(Cs,minDcf,"C", title)


# ---- ___ ----
def c_estimation_poly_eval(Dtr,Ltr,DTE, LTE):
    cs=[1]
    Cs=np.logspace(-5, 2, num=10)
    pca_flag = False
    norm_flag = True
    K = 1
    minDcf=[]
    minDcf_train=[]
    svm_scores=[]

    for i in range(len(cs)):
        for C in Cs:
            if norm_flag:
                Dtr_new, Dts_new = utils.compute_znorm(Dtr, DTE)
            else: 
                Dtr_new=Dtr
                Dts_new=DTE
            if pca_flag:
                P= apply_PCA(Dtr_new,8)
                Dtr_new=np.dot(P.T,Dtr_new)
                Dts_new=np.dot(P.T,Dts_new)
            else:
                Dtr_new=Dtr_new
                Dts_new=Dts_new

            z,H = kernel_poly_H(Dtr_new, Ltr,cs[i],2,K)
            alpha_opt,opt_dual_value= minimize_mod_dual(Dtr_new,C,z,H)
            sv_score,svm_wp=scores_error_kernel_poly(alpha_opt,z,Dtr_new,Dts_new,LTE,cs[i],2,K)
            svm_scores.append(sv_score)
            llrs_train = SVM_kernel_quadratic_training(Dtr_new, Ltr, 5, 1, C, pca_flag, norm_flag)
            minDcf.append(compute_min_DCF(0.5, 1, 10, np.hstack(svm_scores), LTE))
            minDcf_train.append(compute_min_DCF(0.5, 1, 10, np.hstack(llrs_train), Ltr))
            svm_scores=[]
    plotDCF_C_eval(Cs,minDcf, minDcf_train,"C", "svm poly C Evaluation")


# ---- ___ ----
def estimation_gmm_components(D,L, diag, tied, apply_PCA_flag, norm_flag, title):
    Ds =(np.array_split(D, 5, axis=1))
    Ls = (np.array_split(L, 5))
    G0=[1,2,4,8]
    G1=[1,2,4,8]
    Cs=np.logspace(-5, 2, num=10)
    minDcf=[]

    for i in range(len(G0)):
        for g1 in G1:
            gmm_wp=0
            gmm_scores = []

            for j in range(5):
                Dtr,Ltr = np.hstack(Ds[:j]+ Ds[j+1:]), np.hstack(Ls[:j] + Ls[j+1:])
                Dts,Lts = np.asarray(Ds[j]) , np.asarray(Ls[j])

                if norm_flag:
                    Dtr_new, Dts_new = utils.compute_znorm(Dtr, Dts)
                else:
                    Dtr_new=Dtr
                    Dts_new=Dts
                if apply_PCA_flag:
                    P= apply_PCA(Dtr_new,8)
                    Dtr_new=np.dot(P.T,Dtr_new)
                    Dts_new=np.dot(P.T,Dts_new)
                else:
                    Dtr_new=Dtr_new
                    Dts_new=Dts_new
                
                DTR0=Dtr_new[:,Ltr==0]
                DTR1=Dtr_new[:,Ltr==1]
               
                wg0=1.0
                mug0=DTR0.mean(1).reshape((DTR0.shape[0], 1))
                sigmag0 =1/(DTR0.shape[1])*np.dot((DTR0-mug0),(DTR0-mug0).T)
                if diag: sigmag0=sigmag0*np.eye(sigmag0.shape[0])
                U, s, _ = np.linalg.svd(sigmag0)
                s[s<0.01] = 0.01
                C0 = np.dot(U, utils.vcol(s)*U.T)
                updatedGMM0 = ML_GMM_LBG(DTR0, wg0, mug0, C0, G0[i], diag, tied)
                _,score0=logpdf_GMM(Dts_new, updatedGMM0)

                wg1=1.0
                mug1=DTR1.mean(1).reshape((DTR1.shape[0], 1))
                sigmag1 =1/(DTR1.shape[1])*np.dot((DTR1-mug1),(DTR1-mug1).T)
                if diag: sigmag1=sigmag1*np.eye(sigmag1.shape[0])
                U, s, _ = np.linalg.svd(sigmag1)
                s[s<0.01] = 0.01
                C1 = np.dot(U, utils.vcol(s)*U.T)
                updatedGMM1 = ML_GMM_LBG(DTR1, wg1, mug1, C1, g1, diag, tied)
                _,score1=logpdf_GMM(Dts_new, updatedGMM1)

                Sjoint = np.vstack((score0,score1))
                Smarg= utils.vrow(scipy.special.logsumexp(Sjoint, axis=0))
                f=Sjoint-Smarg
                f=np.exp(f)

                gmm_wp += (f.argmax(0)!=Lts).sum()
                diffScores=(score1-score0)[0]
                gmm_scores.append(diffScores)

            gmm_err = gmm_wp/(D.shape[1])
            minDcf.append(compute_min_DCF(0.5, 1, 10, np.hstack(gmm_scores), L))
    plotDCF_gmm_comp(G1,minDcf,"Components",title)


# ---- ___ ----
def estimation_gmm_components_eval(DTR,LTR,DTE,LTE, diag, tied, apply_PCA_flag, norm_flag, title):
    G0=[1,2,4,8]
    G1=[1,2,4,8]
    K = 1
    minDcf=[]
    svm_scores=[]

    for i in range(len(G0)):
        for g1 in G1:
            gmm_wp=0
            gmm_scores = []
            if norm_flag:
                Dtr_new, Dts_new = utils.compute_znorm(DTR, DTE)
            else:
                Dtr_new=DTR
                Dts_new=DTE
            if apply_PCA_flag:
                P= apply_PCA(Dtr_new,8)
                Dtr_new=np.dot(P.T,Dtr_new)
                Dts_new=np.dot(P.T,Dts_new)
            else:
                Dtr_new=Dtr_new
                Dts_new=Dts_new

            DTR0=Dtr_new[:,LTR==0]
            DTR1=Dtr_new[:,LTR==1]
           
            wg0=1.0
            mug0=DTR0.mean(1).reshape((DTR0.shape[0], 1))
            sigmag0 =1/(DTR0.shape[1])*np.dot((DTR0-mug0),(DTR0-mug0).T)
            if diag: sigmag0=sigmag0*np.eye(sigmag0.shape[0])
            U, s, _ = np.linalg.svd(sigmag0)
            s[s<0.01] = 0.01
            C0 = np.dot(U, utils.vcol(s)*U.T)
            updatedGMM0 = ML_GMM_LBG(DTR0, wg0, mug0, C0, G0[i], diag, tied)
            _,score0=logpdf_GMM(Dts_new, updatedGMM0)

            wg1=1.0
            mug1=DTR1.mean(1).reshape((DTR1.shape[0], 1))
            sigmag1 =1/(DTR1.shape[1])*np.dot((DTR1-mug1),(DTR1-mug1).T)
            if diag: sigmag1=sigmag1*np.eye(sigmag1.shape[0])
            U, s, _ = np.linalg.svd(sigmag1)
            s[s<0.01] = 0.01
            C1 = np.dot(U, utils.vcol(s)*U.T)
            updatedGMM1 = ML_GMM_LBG(DTR1, wg1, mug1, C1, g1, diag, tied)
            _,score1=logpdf_GMM(Dts_new, updatedGMM1)

            Sjoint = np.vstack((score0,score1))
            Smarg= utils.vrow(scipy.special.logsumexp(Sjoint, axis=0))
            f=Sjoint-Smarg
            f=np.exp(f)

            gmm_wp += (f.argmax(0)!=LTE).sum()
            diffScores=(score1-score0)[0]
            gmm_scores.append(diffScores)
            minDcf.append(compute_min_DCF(0.5, 1, 10, np.hstack(gmm_scores), LTE))
    plotDCF_gmm_comp(G1,minDcf,"Components",title)
