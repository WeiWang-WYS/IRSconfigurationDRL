"""
Multi-user MIMO, environment
"""

from __future__ import division
import numpy as np
import tensorflow as tf
import math

from math import *

class envMuMIMO:

    def __init__(self, NumAntBS, NumEleIRS, NumUser):
        self.NumAntBS = NumAntBS
        self.NumEleIRS = NumEleIRS
        self.NumUser = NumUser

    def DFT_matrix(self, N_point):
        n, m = np.meshgrid(np.arange(N_point), np.arange(N_point))
        omega = np.exp(-2 * pi * 1j / N_point)
        W = np.power( omega, n * m ) / sqrt(N_point)  ##
        return W

    def CH_Prop(self, H, sigma2, Pilot):
        NumAnt, NumUser = np.shape(H)
        noise = 1/sqrt(2) * (np.random.normal(0, sigma2, size=(NumAnt, NumUser)) + 1j * np.random.normal(0, sigma2, size=(NumAnt, NumUser))) ## Gaussian Noise
        y_rx = np.dot(H, Pilot) + noise
        return y_rx

    def CH_est(self, y_rx, sigma2, Pilot):
        MMSE_matrix = np.matrix.getH(Pilot) / (1+sigma2) ## MMSE channel estimation
        H_est = np.dot(y_rx, MMSE_matrix)
        return H_est

    def Precoding(self, H_est):
        F = np.dot(np.linalg.inv(np.dot(np.matrix.getH(H_est), H_est)), np.matrix.getH(H_est)) ## Zero-forcing Precoding
        NormCoeff = abs(np.diag(np.dot(F, np.matrix.getH(F))))
        NormCoeff = 1/np.sqrt(NormCoeff)
        F =  np.dot(np.diag(NormCoeff), F) ## Normalization
        return F

    def GetRewards(self, Pilot, H_synt, sigma2_BS, sigma2_UE):
        y_rx = self.CH_Prop(H_synt, sigma2_BS, Pilot) ### Received singal
        H_est = self.CH_est(y_rx, sigma2_BS, Pilot)  ### Estimated equivalent channel
        F = self.Precoding(H_est)  ### Zero-Forcing precoding
        H_eq = np.dot(F, H_synt)
        H_eq2 = abs(H_eq * np.conj(H_eq))
        SigPower = np.diag(H_eq2)   #### Singal Power
        IntfPower = H_eq2.sum(axis=0)
        IntfPower = IntfPower - SigPower #### Interference Power
        SINR = SigPower / (IntfPower + sigma2_UE)  ### SNIR
        Rate = np.log2(1+SINR)  #### Data Rate
        return Rate, y_rx, H_est

    def SubSteeringVec(self, Angle, NumAnt):
        SSV = np.exp(1j * Angle * math.pi * np.arange(0, NumAnt, 1))
        SSV = SSV.reshape(-1, 1)
        return SSV

    def ChannelResponse(self, Pos_A, Pos_B, ArrayShape_A, ArrayShape_B):   ## LoS channel response, which is position dependent
        dis_AB = np.linalg.norm(Pos_A - Pos_B)  ## distance
        DirVec_AB = (Pos_A - Pos_B) / dis_AB  ## direction vector
        angleA = [np.linalg.multi_dot([[1, 0, 0], DirVec_AB]), np.linalg.multi_dot([[0, 1, 0], DirVec_AB]),
              np.linalg.multi_dot([[0, 0, 1], DirVec_AB])]
        SteeringVectorA = np.kron(self.SubSteeringVec(angleA[0], ArrayShape_A[0]),
                              self.SubSteeringVec(angleA[1], ArrayShape_A[1]))
        SteeringVectorA = np.kron(SteeringVectorA, self.SubSteeringVec(angleA[2], ArrayShape_A[2]))
        angleB = [np.linalg.multi_dot([[1, 0, 0], DirVec_AB]), np.linalg.multi_dot([[0, 1, 0], DirVec_AB]),
              np.linalg.multi_dot([[0, 0, 1], DirVec_AB])]
        SteeringVectorB = np.kron(self.SubSteeringVec(angleB[0], ArrayShape_B[0]),
                              self.SubSteeringVec(angleB[1], ArrayShape_B[1]))
        SteeringVectorB = np.kron(SteeringVectorB, self.SubSteeringVec(angleB[2], ArrayShape_B[2]))
        H_matrix = np.linalg.multi_dot([SteeringVectorA, np.matrix.getH(SteeringVectorB)])
        return H_matrix


    def H_GenFunLoS(self, Pos_BS, Pos_IRS, Pos_UE, ArrayShape_BS, ArrayShape_IRS, ArrayShape_UE):
        NumUE = len(Pos_UE)
        NumAntBS = np.prod(ArrayShape_BS)
        NumEleIRS = np.prod(ArrayShape_IRS)
        H_BU_LoS = np.zeros((NumAntBS, NumUE)) + 1j * np.zeros((NumAntBS, NumUE))
        H_RU_LoS = np.zeros((NumEleIRS, NumUE)) + 1j * np.zeros((NumEleIRS, NumUE))
        for iu in range(NumUE):
            h_BU_LoS = self.ChannelResponse(Pos_BS, Pos_UE[iu], ArrayShape_BS, ArrayShape_UE)
            H_BU_LoS[:, iu] = h_BU_LoS.reshape(-1)
            h_RU_LoS = self.ChannelResponse(Pos_IRS, Pos_UE[iu], ArrayShape_IRS, ArrayShape_UE)
            H_RU_LoS[:, iu] = h_RU_LoS.reshape(-1)
        H_BR_LoS = self.ChannelResponse(Pos_BS, Pos_IRS, ArrayShape_BS, ArrayShape_IRS)
        return H_BU_LoS, H_BR_LoS, H_RU_LoS


    def H_GenFunNLoS(self, NumAntBS, NumEleIRS, NumUser):
        H_U2B_NLoS = 1 / sqrt(2) * (np.random.normal(0, 1, size=(NumAntBS, NumUser)) + 1j * np.random.normal(0, 1, size=(NumAntBS, NumUser)))
        H_R2B_NLoS = 1 / sqrt(2) * (np.random.normal(0, 1, size=(NumAntBS, NumEleIRS)) + 1j * np.random.normal(0, 1, size=(NumAntBS, NumEleIRS)))
        H_U2R_NLoS = 1 / sqrt(2) * (np.random.normal(0, 1, size=(NumEleIRS, NumUser)) + 1j * np.random.normal(0, 1, size=(NumEleIRS, NumUser)))
        return H_U2B_NLoS, H_R2B_NLoS, H_U2R_NLoS

    def H_syntFun(self, H_U2B, H_R2B, H_U2R, RefVector): ### Syntheize the aggregated wireless channel
        RefPattern_matrix = np.diag(RefVector)
        H_synt = H_U2B +  1*np.linalg.multi_dot([H_R2B, RefPattern_matrix, H_U2R])
        return H_synt