# Development plan
## Explore carla 
- :heavy_check_mark: set up platform
- :heavy_check_mark: asses vehicles, sensors, general environment

## Scenario building
- :heavy_check_mark: explore a way to set up scenario
- :heavy_check_mark: set up first simple straight driving scenario

## Agent Controller
- :heavy_check_mark: explore a way to connnect a control to simulation
- extend python scenarios to handle agents, check where the agent sets up its route gps info, and compare with route approach
    - :heavy_check_mark: make route builder
    - :heavy_check_mark: adapt scenario_runner, add agent1 parameter
    - :heavy_check_mark: make simple slowly forward driving agent
    - :heavy_check_mark: check that all data wanted criteria, input data is available for the agent
    - :heavy_check_mark: add criterias to agent
    - agent needs access to scenario, to assess the criterias for training.
        - the trainer of the agent will use it to compute rewards, terminate the scenario etc.
    - make a detector to check passed waypoints on route
- build a first simple controller 

## First evaluation
- setup SAC controller to learn straight driving
    - have the agent set up the required models, 
    - fill up replay buffer, with adequate (s_t,a_t,s_t+1,r) information
    - monitor the scenario from the trainer 
    - have general logic to initiate scenarios, fill replay buffer, and train on the scenario
- Build way to do imitation learning to build sample buffer for learning 
    - collect (s_t,a_t,s_t+1,r) from npc agent and populate replay buffer
- Test training loop
- Try to learn straight driving

# Next step
- Evaluate setup
- More complex scenarios 

