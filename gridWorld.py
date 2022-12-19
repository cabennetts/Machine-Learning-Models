import numpy as np

# calculates reward and has a flag to check if its for value or policy
def actReward(init, action, flag):
    if init in terminations:
        return init, 0
    reward = rewardSize
    finalPos = np.array(init) + np.array(action)
    if (-1 in finalPos) or (gridSize in finalPos):
        finalPos = init 
    if flag == 0:
        return finalPos, reward
    elif flag == 1:
        return finalPos, -1

# initialization stuff 
rewardSize = -1
gamma = 1
gridSize = 5
terminations = [[0,0],[4,4]]
actions = [[-1,0], [1,0], [0,1], [0,-1]]
maxItr = 2000

def policyIteration():
    print ("#################################")
    print ("##      Policy Iteration       ##")
    print ("#################################")
    # initialization
    vArr = np.zeros((gridSize, gridSize)).round(2)
    states = [[i,j] for i in range(gridSize) for j in range(gridSize)]
    print("Iteration 0: Initial values")
    print(vArr)
    print("-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")
    for itr in range(maxItr):
        prevArr = np.copy(vArr).round(2)
        for s in states:
            weightedRs = 0
            for a in actions:
                finalPos, reward = actReward(s, a, 0)
                weightedRs += (1/len(actions))*(reward+(gamma*vArr[finalPos[0], finalPos[1]]))
            prevArr[s[0], s[1]] = weightedRs
        compare = vArr == prevArr
        if compare.all():
            print("Iteration ", itr+1, ": Final Iteration")
            print(vArr.round(2))
            print("-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")
            break
        vArr = prevArr
        if itr in [0,9]:
            print("Iteration {}".format(itr+1))
            print(vArr.round(2))
            print("-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")

def valueIteration():
    print ("#################################")
    print ("##       Value Iteration       ##")
    print ("#################################")
    vArr = np.zeros((gridSize, gridSize)).round(2)
    states = [[i,j] for i in range(gridSize) for j in range(gridSize)]
    print("Iteration 0: Initial values")
    print(vArr)
    print("-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")
    for itr in range(maxItr):
        prevArr = np.copy(vArr).round(2)
        for s in states:
            weightedRs = np.empty(shape=0)
            for a in actions:
                finalPos, reward = actReward(s, a, 1)
                weightedR = reward +vArr[finalPos[0], finalPos[1]]
                weightedRs = np.insert(weightedRs, weightedRs.size, weightedR)
            prevArr[s[0], s[1]] = np.max(weightedRs)
        compare = vArr == prevArr
        if compare.all():
            print("Iteration ", itr+1, ": Final Iteration")
            print(vArr.round(2))
            print("-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")
            break
        # this is to print iteration 2
        vArr = prevArr
        if itr in [0,1]:
            print("Iteration ", itr+1)
            print(vArr.round(2))
            print("-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")

# run both functions
policyIteration()
valueIteration()