# [Graded]
def argmax(q_values):
    """
    Takes in a list of q_values and returns the index
    of the item with the highest value. Breaks ties randomly.
    returns: int - the index of the highest value in q_values
    """
    top = float("-inf")
    ties = []
    
    for i in range(len(q_values)):
        # if a value in q_values is greater than the highest value, then update top and reset ties to zero
        # if a value is equal to top value, then add the index to ties (hint: do this no matter what)
        # Note: You do not have to follow this exact solution. You can choose to do your own implementation.
        ### START CODE HERE ###
        if(q_values[i]>top):
            top = q_values[i]
            ties = []
            ties.append(i)
        elif(q_values[i]==top):
            ties.append(i)
        ### END CODE HERE ###
    
    # return a random selection from ties. (hint: look at np.random.choice)
    ### START CODE HERE ###
    selected = np.random.choice(ties, size=1, replace=False, p=None)
    
    ### END CODE HERE ###
    
    return selected # change this



# Greedy agent here [Graded]
class GreedyAgent(main_agent.Agent):
    def agent_step(self, reward, observation):
        """
        Takes one step for the agent. It takes in a reward and observation and 
        returns the action the agent chooses at that time step.
        
        Arguments:
        reward -- float, the reward the agent received from the environment after taking the last action.
        observation -- float, the observed state the agent is in. Do not worry about this for this assignment 
        as you will not use it until future lessons.
        Returns:
        current_action -- int, the action chosen by the agent at the current time step.
        """
        ### Useful Class Variables ###
        # self.q_values : An array with the agent’s value estimates for each action.
        # self.arm_count : An array with a count of the number of times each arm has been pulled.
        # self.last_action : The action that the agent took on the previous time step.
        #######################
        
        # Update action values. Hint: Look at the algorithm in section 2.4 of the textbook.
        # Increment the counter in self.arm_count for the action from the previous time step
        # Update the step size using self.arm_count
        # Update self.q_values for the action from the previous time step
        # (~3-5 lines)
        ### START CODE HERE ###
        #self.arm_count[self.last_action] = self.arm_count[self.last_action]+1
        self.last_action = int(self.last_action)
        self.arm_count[self.last_action]+=1
        self.q_values[self.last_action] = self.q_values[self.last_action] + 1/(self.arm_count[self.last_action])*(reward -self.q_values[self.last_action])
        ### END CODE HERE ###
        
        # current action = ? # Use the argmax function you created above
        # (~2 lines)
        ### START CODE HERE ###
        current_action = argmax(self.q_values)
        ### END CODE HERE ###
    
        self.last_action = current_action
        
        return current_action
        

# Epsilon Greedy Agent here [Graded]
class EpsilonGreedyAgent(main_agent.Agent):
    def agent_step(self, reward, observation):
        """
        Takes one step for the agent. It takes in a reward and observation and 
        returns the action the agent chooses at that time step.
        
        Arguments:
        reward -- float, the reward the agent received from the environment after taking the last action.
        observation -- float, the observed state the agent is in. Do not worry about this for this assignment 
        as you will not use it until future lessons.
        Returns:
        current_action -- int, the action chosen by the agent at the current time step.
        """
        
        ### Useful Class Variables ###
        # self.q_values : An array with the agent’s value estimates for each action.
        # self.arm_count : An array with a count of the number of times each arm has been pulled.
        # self.last_action : The action that the agent took on the previous time step.
        # self.epsilon : The probability an epsilon greedy agent will explore (ranges between 0 and 1)
        #######################
        
        # Update action-values - this should be the same update as your greedy agent above
        # (~3-5 lines)
        ### START CODE HERE ###
        self.last_action = int(self.last_action)
        self.arm_count[self.last_action]+=1
        self.q_values[self.last_action] = self.q_values[self.last_action] + 1/(self.arm_count[self.last_action])*(reward -self.q_values[self.last_action])
        ### END CODE HERE ###
        
        # Choose action using epsilon greedy
        # Randomly choose a number between 0 and 1 and see if it is less than self.epsilon
        # (Hint: look at np.random.random()). If it is, set current_action to a random action.
        # Otherwise choose current_action greedily as you did above.
        # (~4 lines)
        ### START CODE HERE ###
        random_choice_prob = np.random.random(1)
        if(random_choice_prob<self.epsilon):
            current_action = np.random.randint(0,len(self.q_values))
        elif(random_choice_prob>=self.epsilon):
            current_action = argmax(self.q_values)
        ### END CODE HERE ###
        
        
        self.last_action = current_action
        
        return current_action


# Constant Step Size Agent Here [Graded]
# Greedy agent here
class EpsilonGreedyAgentConstantStepsize(main_agent.Agent):
    def agent_step(self, reward, observation):
        """
        Takes one step for the agent. It takes in a reward and observation and 
        returns the action the agent chooses at that time step.
        
        Arguments:
        reward -- float, the reward the agent received from the environment after taking the last action.
        observation -- float, the observed state the agent is in. Do not worry about this for this assignment 
        as you will not use it until future lessons.
        Returns:
        current_action -- int, the action chosen by the agent at the current time step.
        """
        
        ### Useful Class Variables ###
        # self.q_values : An array with the agent’s value estimates for each action.
        # self.arm_count : An array with a count of the number of times each arm has been pulled.
        # self.last_action : The action that the agent took on the previous time step.
        # self.step_size : A float which is the current step size for the agent.
        # self.epsilon : The probability an epsilon greedy agent will explore (ranges between 0 and 1)
        #######################
        
        # Update q_values for action taken at previous time step 
        # using self.step_size intead of using self.arm_count
        # (~1-2 lines)
        ### START CODE HERE ###
        self.last_action = int(self.last_action)
        self.arm_count[self.last_action]+=1
        self.q_values[self.last_action] = self.q_values[self.last_action] + self.step_size*(reward -self.q_values[self.last_action])
        ### END CODE HERE ###
        
        # Choose action using epsilon greedy. This is the same as you implemented above.
        # (~4 lines)
        ### START CODE HERE ###
        random_choice_prob = np.random.random(1)
        if(random_choice_prob<self.epsilon):
            current_action = np.random.randint(0,len(self.q_values))
        elif(random_choice_prob>=self.epsilon):
            current_action = argmax(self.q_values)
        ### END CODE HERE ###
        
        self.last_action = current_action
        
        return current_action