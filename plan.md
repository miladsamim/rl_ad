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
    - :heavy_check_mark: agent needs access to scenario, to assess the criterias for training.
        - the trainer of the agent will use it to compute rewards, terminate the scenario etc.
    - :heavy_check_mark: make a detector to check passed waypoints on route
- :heavy_check_mark: build a first simple controller 

## First evaluation
- Set up CarlEnv wrapper to simulate gym api 
    - detect whether reload is necessary when restarting 
    - :heavy_check_mark connect with MPI 
    - reward computation
        - :heavy_check_mark: criteria processing
    - :heavy_check_mark: state processing 
- Set up trainer
    - Trainer should select scenarios, weather, start the agent, then it should simply execute for number of specified learning steps, and then sample scenario on reset, log metrics, and manage the buffer, perhaps store buffer (or at least for npc-agents maybe) 
- setup SAC controller to learn straight driving
    - have the agent set up the required models, 
    - fill up replay buffer, with adequate (s_t,a_t,s_t+1,r) information
    - monitor the scenario from the trainer 
    - have general logic to initiate scenarios, fill replay buffer, and train on the scenario
- Build way to do imitation learning to build sample buffer for learning 
    - collect (s_t,a_t,s_t+1,r) from npc agent and populate replay buffer
- Test training loop
- Try to learn straight driving

# First Evaluation 1
- :heavy_check_mark: finalize reward computation
    - $R_t = -0.1 + 1*rc - 2*wl - 200*c -1*d - 10*of$
    - rc (route completion) = positive every driven meter on route
    - wl (wrong lane) = penalty for crossing in to wrong lane 
    - c (collision) = penalty for colliding into objects (cars, pedestrians, buildings, etc.)
    - d (deviation from center of lane) = penalty for not being at the center of lane 
    - of (off-road) = penalty for being off-road sidewalk, grass, etc.
- :heavy_check_mark: checklog wanted data and training measures
- :heavy_check_mark: set up a set of scenarios to train on for 3M steps
- :heavy_check_mark: start training 

# Preparation for Test 1
- :heavy_check_mark: new Memory Buffer
- :heavy_check_mark: Set up model
    - check model trains, evals
    - check that state, actions matches temporally 
- :heavy_check_mark: Check that npc actions are matched well
- Train for StraightDriving
- Add scenarios

# Next step
- Evaluate setup
- More complex scenarios