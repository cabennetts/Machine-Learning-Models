import numpy as np
import random
#
# FUNCTIONS MUST BE RUN ONE AT A TIME (AT BOTTOM OF FILE):(
# FUNCTIONS MUST BE RUN ONE AT A TIME (AT BOTTOM OF FILE) :(
# FUNCTIONS MUST BE RUN ONE AT A TIME (AT BOTTOM OF FILE) :(
# FUNCTIONS MUST BE RUN ONE AT A TIME (AT BOTTOM OF FILE) :(    
# FUNCTIONS MUST BE RUN ONE AT A TIME (AT BOTTOM OF FILE) :(
# FUNCTIONS MUST BE RUN ONE AT A TIME (AT BOTTOM OF FILE) :(
# FUNCTIONS MUST BE RUN ONE AT A TIME (AT BOTTOM OF FILE) :(

# Helper functions
def isTermination(pos): 
    #determine if pos is termination state
    return any(np.array_equal(x, pos) for x in terminations)

def actReward(init, action):
    r = rewardSize
    finalPos = np.array(init) + np.array(action)
    if -1 in finalPos or gridSize in finalPos:
        finalPos = init
    if isTermination(finalPos):
        r=0
    return finalPos, r

def getPos(state):
    # get position from a state value
    return [state//gridSize, state%gridSize]

def stateVal(pos):
    # return value from a position
    return pos[0]*gridSize + pos[1]

def printMatrices(arr):
    print("##### S/A #####")
    print(end="\t")
    for i in range(len(arr)):
        print(i, end="\t")
    print("")
    for i in range(len(arr)):
        print(i, end="\t")
        for j in range(len(arr[0])):
            print(arr[i][j], end="\t")
        print("")

def isConverged(Q_copy, Q, totalStates, R):
    for i in range(1,totalStates-1):
        for j in range(0,totalStates):
            if R[i][j] != -1 and Q[i][j] != 0: 
                break 
            if j == totalStates-1:
                return False
    compare = Q == Q_copy
    return compare.all()

# Initialization's
rewardSize = -1
gamma = 0.9
gridSize = 5
terminations = np.array([[0,0],[4,4]])
actions = np.array([[-1,0], [1,0], [0,1], [0,-1]])
maxItr = 5000

# PART 1: RL Monte Carlo First Visit Algorithm
def mcFirstVisit():
    print ("##################################")
    print ("##   Monte Carlo First Visit    ##")
    print ("##################################")
    states = [[i, j] for i in range(gridSize) for j in range(gridSize)]
    Ns = np.zeros((5,5))
    Ss = np.zeros((5,5))
    Vs = np.zeros((5,5))
    print("Episode 0: Initial Values (First Visit)\n")
    print("N(s)\n", Ns)
    print("S(s)\n", Ss)
    print("V(s)\n", Vs)

    for i in range(1, maxItr):
        Vs_copy = np.copy(Vs)
        pos = states[random.randint(1, len(states)-2)]
        GsTable = np.zeros((0,5))
        visited = np.zeros((5,5))
        k = 1
        row = [k, stateVal(pos), 0 if isTermination(pos) else rewardSize, gamma, 0]
        GsTable = np.vstack([GsTable, row])

        while not isTermination(pos):
            k += 1
            action = random.choice(actions)
            pos, reward = actReward(pos, action)
            row = [k, stateVal(pos), reward, gamma, 0]
            GsTable = np.vstack([GsTable, row])
        
        totalActions = GsTable.shape[0]

        for i in range(totalActions-1):
            k = int(GsTable[i][0])
            s = int(GsTable[i][1])
            state = getPos(s)
            for j in range(int(totalActions-k) + 1):
                curr = GsTable[int(k + j -1)]
                GsTable[i][4] += pow(curr[3], j)*(curr[2])
            
            if not isTermination(state) and not visited[state[0]][state[1]]:
                Ns[state[0]][state[1]] += 1
                Ss[state[0]][state[1]] += GsTable[i][4]
            visited[state[0]][state[1]] = 1
        
        for s in range(gridSize * gridSize):
            state = getPos(s)
            if int(Ns[state[0]][state[1]]) > 0:
                Vs[state[0]][state[1]] = Ss[state[0]][state[1]] / int(Ns[state[0]][state[1]])

        Vs = np.around(Vs, decimals=2)
        Ss = np.around(Ss, decimals=2)
        GsTable = np.around(GsTable, decimals=2)
        compare = Vs == Vs_copy
        if compare.all():
            print("Episode ", i, ": Final Episode (First Visit) CONVERGED")
            print("N(s)\n", Ns)
            print("S(s)\n", Ss)
            print("V(s)\n", Vs)
            print("k, s, r, v, G(s)")
            print(GsTable)
            print("\n")
            exit()
        if i in [1,10]:
            print("Episode ", i, ": (First Visit)")
            print("N(s)\n", Ns)
            print("S(s)\n", Ss)
            print("V(s)\n", Vs)
            print("k, s, r, v, G(s)")
            print(GsTable)
            print("\n")
        if i == maxItr-1:
            print("Episode ", i, ": Final Episode (First Visit) Unable to converge!")
            print("N(s)\n", Ns)
            print("S(s)\n", Ss)
            print("V(s)\n", Vs)
            exit()

# PART 2: RL Monte Carlo Every Visit Algorithm
def mcEveryVisit():
    print ("##################################")
    print ("##   Monte Carlo Every Visit    ##")
    print ("##################################")
    states = [[i, j] for i in range(gridSize) for j in range(gridSize)]
    Ns = np.zeros((gridSize, gridSize))
    Ss = np.zeros((gridSize, gridSize))
    Vs = np.zeros((gridSize, gridSize))
    print("Episode 0 (Initial Values - Every Visit Method)")
    print("N(s)\n", Ns)
    print("S(s)\n", Ss)
    print("V(s)\n", Vs)

    for e in range(1,maxItr):
        Vs_copy = np.copy(Vs)
        pos = states[random.randint(1,len(states)-2)]
        GsTable = np.zeros((0, 5))
        k = 1
        row = [k, stateVal(pos), 0 if isTermination(pos) else rewardSize, gamma, 0]
        GsTable = np.vstack([GsTable,row])

        while not isTermination(pos):
            k += 1
            action = random.choice(actions) 
            pos, reward = actReward(pos, action)
            row = [k,stateVal(pos), reward, gamma, 0]
            GsTable = np.vstack([GsTable,row])

        totalActions = GsTable.shape[0]
        for i in range(totalActions):
            k = int(GsTable[i][0])
            s = int(GsTable[i][1])
            state = getPos(s)
            for j in range(int(totalActions-k) + 1):
                curr = GsTable[int(k + j - 1)]
                GsTable[i][4] += pow(curr[3], j)*(curr[2])
            if not isTermination(state):
                Ns[state[0]][state[1]] += 1
            Ss[state[0]][state[1]] += GsTable[i][4]

        for s in range(gridSize*gridSize):
            state = getPos(s)
            if int(Ns[state[0]][state[1]]) > 0:
                Vs[state[0]][state[1]] = Ss[state[0]][state[1]]/int(Ns[state[0]][state[1]])
        
        Vs = np.around(Vs, decimals=2)
        Ss = np.around(Ss, decimals=2)
        compare = Vs == Vs_copy
        if compare.all():
            print("Episode ", e, ": Final Episode (Every Visit) CONVERGED")
            print("N(s)\n", Ns)
            print("S(s)\n", Ss)
            print("V(s)\n", Vs) 
            print("k, s, r, γ, G(s) values:")
            print(GsTable)
            exit()
        if e in [1,10]:
            print("Episode ", e, ": (Every Visit)")
            print("N(s)\n", Ns)
            print("S(s)\n", Ss)
            print("V(s)\n", Vs) 
            print("k, s, r, γ, G(s) values:")
            print(GsTable)
        if e == maxItr-1:
            print("Episode ", e, ": Final Episode (Every Visit) Unable to converge!")
            print("N(s)\n", Ns)
            print("S(s)\n", Ss)
            print("V(s)\n", Vs) 
            exit()

# PART 3: RL Q-Learning Algorithm
def qLearning():
    print ("##################################")
    print ("##          Q-Learning          ##")
    print ("##################################")
    actions = np.array([[-1, 0], [1, 0], [0, 1], [0, -1]])
    totalStates = pow(gridSize, 2)
    Q = np.zeros((totalStates, totalStates))
    R = np.full((totalStates, totalStates), -1)

    for s in range(gridSize * gridSize):
        if isTermination(getPos(s)):
            R[int(s)][int(s)] = 100
        for action in actions:
            neighbor = np.array(getPos(s)) + np.array(action)
            if not(-1 in neighbor or gridSize in neighbor): 
                R[int(s)][stateVal(neighbor)] = 100 if isTermination(neighbor) else 0

    print("Q-Learning Rewards")
    printMatrices(R)
    print("Episode 0: Values (Initial Values - Q-Learning Method)")
    printMatrices(Q)
    for episode in range(1, maxItr):
        Q_copy = np.copy(Q)
        state = random.randint(0, totalStates-1)

        while not isTermination(getPos(state)):
            max = 0
            actions = np.array([])
            for i in range(totalStates):
                if R[state][i] != -1:
                    actions = np.append(actions, i)
            action = int(random.choice(actions))
            nextAct = np.array([])
            for i in range(totalStates):
                    if R[action][i] != -1:
                        nextAct = np.append(nextAct, i)
            for next in nextAct:
                if Q[action][int(next)] > max:
                    max = Q[action][int(next)]
            Q[state][action] = R[state][action] + (gamma * max)
            state = action

        Q = np.around(Q, decimals=2)
        if isConverged(Q_copy, Q, totalStates, R):
            print("Episode {} : Final Episode (Q-Learning) CONVERGED".format(episode))
            printMatrices(Q)
            exit()
        if episode in [1,10]:
            print("Episode {} : (Q-Learning)".format(episode))
            printMatrices(Q)
        if episode == maxItr-1:
            print("Episode {} : Final Episode (SARSA) Unable to converge!".format(episode))

# PART 4: RL SARSA Algorithm
def sarsa():
    print ("##################################")
    print ("##            SARSA             ##")
    print ("##################################")
    actions = np.array([[-1, 0], [1, 0], [0, 1], [0, -1]])
    totalStates = pow(gridSize, 2)
    Q = np.zeros((totalStates, totalStates))
    R = np.full((totalStates, totalStates), -1)

    for s in range(gridSize * gridSize):
        if isTermination(getPos(s)):
            R[int(s)][int(s)] = 100
        for action in actions:
            neighbor = np.array(getPos(s)) + np.array(action)
            if not(-1 in neighbor or gridSize in neighbor): 
                R[int(s)][stateVal(neighbor)] = 100 if isTermination(neighbor) else 0

    print("SARSA Rewards")
    printMatrices(R)
    print("Episode 0: Values (Initial Values - SARSA Method)")
    printMatrices(Q)
    for episode in range(1, maxItr):
        Q_copy = np.copy(Q)
        state = random.randint(0, totalStates-1)
        max = 0
        while not isTermination(getPos(state)):
            actions = np.array([])
            for i in range(totalStates):
                if R[state][i] != -1:
                    actions = np.append(actions, i)
            nextAct = np.array([])
            nextAct = np.append(nextAct, int(actions[0]))
            for i in np.delete(action, 0):
                if Q[state][int(i)] == Q[state][int(nextAct[0])]:
                    nextAct = np.append(nextAct, i)
                if Q[state][int(i)] > Q[state][int(nextAct[0])]:
                    nextAct = np.array([i])
            action = int(random.choice(nextAct))
            nextAct = np.array([])
            for i in range(totalStates):
                if R[action][i] != 1:
                    nextAct = np.append(nextAct, i)
            for next in nextAct:
                if Q[action][int(next)] > max:
                    max = Q[action][int(next)]
            Q[state][action] = R[state][action] + (gamma * max)
            state = action

        Q = np.around(Q, decimals=2)
        if isConverged(Q_copy, Q, totalStates, R):
            print("Episode {} : Final Episode (SARSA) CONVERGED".format(episode))
            printMatrices(Q)
            exit()
        if episode in [1,10]:
            print("Episode {} : (SARSA)".format(episode))
            printMatrices(Q)
        if episode == maxItr-1:
            print("Episode {} : Final Episode (SARSA) Unable to converge!".format(episode))



# PART 5: RL Decaying Epsilon-Greedy Algorithm
def decayEpsilonGreedy():
    print ("##################################")
    print ("##   Decaying Epsilon-Greedy    ##")
    print ("##################################")


# PART 6: Cumulative Average Reward Comparison
def cumAvgRewardCompare():
    print ("##################################")
    print ("##    Cumulative Avg Reward     ##")
    print ("##################################")


mcFirstVisit()
# mcEveryVisit()
# qLearning()
# sarsa()