"""
Title: Intelligent Reflecting Surface Configurations for Smart Radio Using Deep Reinforcement Learning
Author Contacts: richardwangseu@gmail.com
"""

import numpy as np
import matplotlib.pyplot as plt
from MuMIMOClass import *
from DQN import *
from MAB import *

if __name__ == "__main__":

## Simulation Parameters
    EPISODES = 2000
    NumAntBS = 2
    NumEleIRS = 32
    NumUser = 2
    sigma2_BS = .1  # Noise level at BS side
    sigma2_UE = .5  # Noise level at UE side
    Pos_BS = np.array([0, 0, 10])  # Position of BS
    Pos_IRS = np.array([-2, 5, 5])  # Position of IRS
    MuMIMO_env = envMuMIMO(NumAntBS, NumEleIRS, NumUser) # Environment
    batch_size = 8
    state_size = [NumAntBS*NumUser*2, NumEleIRS * 2]
    QuantLevel = 8  # Quantization level of Phase shift

## Action Set
    ShiftCodebook = [np.exp(1j * pi * 2 * np.arange(0, NumEleIRS, 1) / NumEleIRS),
                     np.exp(-1j * pi * 2 * np.arange(0, NumEleIRS, 1) / NumEleIRS),
                     np.exp(3j * pi * 2 * np.arange(0, NumEleIRS, 1) / NumEleIRS),
                     np.exp(-3j * pi * 2 * np.arange(0, NumEleIRS, 1) / NumEleIRS),
                     np.exp(0j * pi * 2 * np.arange(0, NumEleIRS, 1) / NumEleIRS)]
    ShiftCodebook = np.array(ShiftCodebook)
    action_size = np.size(ShiftCodebook, 0)

## Channel Dynamics
    block_duration = 1  ### When block_duration>1, ESC will be applied
    BlockPerEpi = 20
    TimeTotal = BlockPerEpi * block_duration

## DDQN
    agent = DQNAgent(state_size, action_size)

## MAB
    MABagent = MAB(NumEleIRS)

## Initialization
    RateExCount = np.zeros(action_size)
    Rate_DQN_seq_episode = np.zeros(EPISODES)
    Rate_Random_seq_episode = np.zeros(EPISODES)
    Rate_MAB_seq_episode= np.zeros(EPISODES)

    RefVector = np.exp(1j * pi * np.zeros((1, NumEleIRS)))
    RefVector_bench_random = RefVector
    Pilot = MuMIMO_env.DFT_matrix(NumUser)  ## Plot pattern
    ArrayShape_BS = [NumAntBS, 1, 1]  ## array shape
    ArrayShape_IRS = [1, NumEleIRS, 1]  ##
    ArrayShape_UE = [1, 1, 1]  ## UE is with 1 antenna

    Rate_Random_seq_block = np.zeros(BlockPerEpi)
    Rate_DQN_seq_block = np.zeros(BlockPerEpi)
    Rate_MAB_seq_block = np.zeros(BlockPerEpi)

###########################################
    for epi in range(EPISODES):
        Pos_UE = np.array([[np.random.rand(1) * 10, np.random.rand(1) * 10, 1.5], [np.random.rand(1) * 10, np.random.rand(1) * 10, 1.5]], dtype = np.float)  ## UE positions are randomly generated in each episode
        H_U2B_LoS, H_R2B_LoS, H_U2R_LoS = MuMIMO_env.H_GenFunLoS(Pos_BS, Pos_IRS, Pos_UE, ArrayShape_BS, ArrayShape_IRS, ArrayShape_UE)  ## LoS component
        SumRate_seq = np.zeros(block_duration) ### Check the performance of ESC

        for block in range(BlockPerEpi):
            H_U2B_NLoS, H_R2B_NLoS, H_U2R_NLoS = MuMIMO_env.H_GenFunNLoS(NumAntBS, NumEleIRS, NumUser)
            K = 10 ## K-factor
            H_U2B = sqrt(1 / (K + 1)) * H_U2B_NLoS + sqrt(K / (K + 1)) * H_U2B_LoS
            H_R2B = sqrt(1 / (K + 1)) * H_R2B_NLoS + sqrt(K / (K + 1)) * H_R2B_LoS
            H_U2R = sqrt(1 / (K + 1)) * H_U2R_NLoS + sqrt(K / (K + 1)) * H_U2R_LoS
            H_synt = MuMIMO_env.H_syntFun(H_U2B, H_R2B, H_U2R, RefVector[0])  ### The aggregated wireless channel

####################################################################################
            DFTcodebook = sqrt(NumEleIRS) * MuMIMO_env.DFT_matrix(NumEleIRS)

### Benchmark: Random Reflection
            random_index = random.randrange(len(DFTcodebook))
            RefVector_bench_random = DFTcodebook[random_index, :]
            H_synt_bench = MuMIMO_env.H_syntFun(H_U2B, H_R2B, H_U2R, RefVector_bench_random)
            Rate_bench, _, _ = MuMIMO_env.GetRewards(Pilot, H_synt_bench, sigma2_BS, sigma2_UE)
            random_rate = sum(Rate_bench)
            Rate_Random_seq_block[block] = random_rate

### Benchmark: Multi-arm bandit
            act_index = MABagent.act_sel()
            RefVector_bench_MAB = DFTcodebook[act_index, :]
            H_synt_bench = MuMIMO_env.H_syntFun(H_U2B, H_R2B, H_U2R, RefVector_bench_MAB)
            Rate_bench, _, _ = MuMIMO_env.GetRewards(Pilot, H_synt_bench, sigma2_BS, sigma2_UE)
            MAB_rate = sum(Rate_bench)
            MABagent.Q_update(act_index, MAB_rate)
            Rate_MAB_seq_block[block] = MAB_rate
### Benchmark Ends


############################# Current State
            if block==0:
                Rate, y_rx, H_est = MuMIMO_env.GetRewards(Pilot, H_synt, sigma2_BS, sigma2_UE)
                H_est_vector = np.reshape(H_est, (1, NumAntBS * NumUser))
                Current_State = [np.concatenate((H_est_vector.real, H_est_vector.imag), axis=1), np.concatenate((RefVector.real, RefVector.imag), axis=1)]
            else:
                Current_State = Next_State

############################# Action
            Flag = 1 ## Flag for ESC
            for i_time in range(block_duration):
                Rate, y_rx, H_est = MuMIMO_env.GetRewards(Pilot, H_synt, sigma2_BS, sigma2_UE)
                H_est_vector = np.reshape(H_est, (1, NumAntBS * NumUser))
                if i_time == 0: ## Coarse phase shift
                    if epi == 0:
                        action = random.randrange(len(ShiftCodebook))
                        act_type = 'random'
                    else:
                        action, act_type = agent.act(Current_State)
                    RefVector = RefVector * ShiftCodebook[action, :] ## Action: Absolute phase shift
                    H_synt = MuMIMO_env.H_syntFun(H_U2B, H_R2B, H_U2R, RefVector[0])
                    Rate, y_rx, H_est = MuMIMO_env.GetRewards(Pilot, H_synt, sigma2_BS, sigma2_UE)

                ### Estimate the rate, exclusively used in ESC
                    Rate_est, _, _ = MuMIMO_env.GetRewards(Pilot, H_est, sigma2_BS, sigma2_UE)
                    SumRate_seq[i_time] = sum(Rate_est) ##

                else:  ## Fine Phase Shift  -- Dither
                    if Flag == 1:  ## When Flag == 1, generate a dither
                        ### Dither based search
                        Dither = np.exp(1j * 2 * pi * 1 / (2 ** QuantLevel) * (np.random.randint(8, size=(1, NumEleIRS)) - 4)) ## a small-scale dither
                        RefVector = RefVector * Dither[0]
                        H_synt = MuMIMO_env.H_syntFun(H_U2B, H_R2B, H_U2R, RefVector[0])
                        Rate, y_rx, H_est = MuMIMO_env.GetRewards(Pilot, H_synt, sigma2_BS, sigma2_UE)
                        ######################## Estimate the rate
                        Rate_est, _, _ = MuMIMO_env.GetRewards(Pilot, H_est, sigma2_BS, sigma2_UE)
                        SumRate_seq[i_time] = sum(Rate_est)  ## Estimated Performance
                        #############
                        if SumRate_seq[i_time] > SumRate_seq[i_time - 1]:
                            Flag = 1
                        else:
                            Flag = -1

                    else:  ## When Flag == -1, generate the phase shift vector that is opposite to the dither, i.e., np.conj(Dither[0]) * np.conj(Dither[0])
                        RefVector = RefVector * np.conj(Dither[0]) * np.conj(Dither[0])
                        H_synt = MuMIMO_env.H_syntFun(H_U2B, H_R2B, H_U2R, RefVector[0])
                        Rate, y_rx, H_est = MuMIMO_env.GetRewards(Pilot, H_synt, sigma2_BS, sigma2_UE)
                        ######################## Estimate the rate
                        Rate_est, _, _ = MuMIMO_env.GetRewards(Pilot, H_est, sigma2_BS, sigma2_UE)
                        SumRate_seq[i_time] = sum(Rate_est)
                        Flag = 1

            H_synt = MuMIMO_env.H_syntFun(H_U2B, H_R2B, H_U2R, RefVector[0])
            Rate, y_rx, H_est = MuMIMO_env.GetRewards(Pilot, H_synt, sigma2_BS, sigma2_UE)
            Rate_DQN_seq_block[block] = sum(Rate)   ## Performance feedback

############################# Reward
            if Rate_DQN_seq_block[block] > 10:  ## Threshold -- 10
                Reward = Rate_DQN_seq_block[block]
            else:
                Reward = Rate_DQN_seq_block[block] - 100 ## Penalty

############################# Next State
            H_est_vector = np.reshape(H_est, (1, NumAntBS * NumUser))
            Next_State = [np.concatenate((H_est_vector.real, H_est_vector.imag), axis=1), np.concatenate((RefVector.real, RefVector.imag), axis=1)]

############################# Memorize
            agent.memorize(Current_State, action, Reward, Next_State)
            if len(agent.memory) > batch_size:
                loss = agent.replay(batch_size)

##### Output: Moving average
        N = 64
        if epi>=N:
            print(
                "episode: {}, e: {:.2}, MovingAveRLearning:{:.4f}, MovingAveRandom:{:.4f}, MovingAveMAB:{:.4f}".format(
                    epi, agent.epsilon, np.mean(Rate_DQN_seq_episode[epi-N:epi]), np.mean(Rate_Random_seq_episode[epi-N:epi]), np.mean(Rate_MAB_seq_episode[epi-N:epi]))
            )
        else:
            print("episode: {}, e: {:.2}".format(epi, agent.epsilon) )

        if epi % 20 == 0: ################## Update target model
            agent.update_target_model()
            agent.save("./IRS_DQN.h5")

        Rate_DQN_seq_episode[epi] = np.mean(Rate_DQN_seq_block)
        Rate_Random_seq_episode[epi] = np.mean(Rate_Random_seq_block)
        Rate_MAB_seq_episode[epi] = np.mean(Rate_MAB_seq_block)


##########################  PLOT
    import pandas as pd
    import time
    localtime = time.localtime(time.time())
    print(localtime)

    dataframe = pd.DataFrame({'Rate_DQN_seq_episode': Rate_DQN_seq_episode,
                              'Rate_Random_seq_episode': Rate_Random_seq_episode,
                             'Rate_MAB_seq_episode': Rate_MAB_seq_episode})
    dataframe.to_csv( "Rate"  + str(localtime)  +  ".csv", index=False, sep=',')

    def get_moving_average(mylist, N):
        cumsum, moving_aves = [0], []
        for i, x in enumerate(mylist, 1):
            cumsum.append(cumsum[i - 1] + x)
            if i >= N:
                moving_ave = (cumsum[i] - cumsum[i - N]) / N
                moving_aves.append(moving_ave)
        return moving_aves

    fig = plt.figure()
    plt.xlabel('Time', fontsize=14)
    plt.ylabel('Rate', fontsize=14, labelpad=-2)

    SumRate_seq_ave = get_moving_average(Rate_DQN_seq_episode, N)
    RandomRate_seq_ave = get_moving_average(Rate_Random_seq_episode, N)
    MABRate_seq_ave = get_moving_average(Rate_MAB_seq_episode, N)

    plt.figure()
    x = np.arange(len(SumRate_seq_ave)) + N
    plt.plot(x, SumRate_seq_ave, 'r-', linewidth=3)
    plt.plot(x, RandomRate_seq_ave, 'g-', linewidth=3)
    plt.plot(x, MABRate_seq_ave, 'k-', linewidth=3)
    plt.xlabel("Episode")
    plt.ylabel("Performance $P_m$ (bps/Hz)")
    plt.legend(['ReLearning', 'Random','MAB'])
    plt.savefig('destination_path.eps', format='eps')
    plt.savefig('destination_path.pdf')
    plt.show()
