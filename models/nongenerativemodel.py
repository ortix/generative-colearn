import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class NonGenerativeModel:

    def __init__(self, config=None):
        self._setConfig(config)
        self._makePredictionModel()
        self._makeLearningModel()

    def predict(self, initial_state, final_state):
        return self.predictionModel.predict([initial_state, final_state])

    def train(self, data):
        self.learningModel.fit(data, epochs= self.epochs, batch_size=self.batch_size) 

    def load(self, fileName):
        self.learningModel.load_weights(fileName)

    def save(self, fileName):
        self.learningModel.save_weights(fileName)
        

    def _setConfig(self, config):
        if config is None:
            self.dof = 1
            self.layerSize = 32
            self.maxTime = 2.0
            self.maxCost = 2.0
            self.maxStates = [1.0, 1.0]
            self.numLayers = 3
            self.epochs = 200
            self.batch_size = 64
        else:
            self.dof = config["dof"]
            self.layerSize = config["layerSize"]
            self.numLayers = config["numLayers"]
            self.maxTime = config["maxTime"]
            self.maxCost = config["maxCost"]
            self.maxStates = config["maxStates"]
            self.numLayers = config["numLayers"]
            self.epochs = config["epochs"]
            self.batch_size = config["batch_size"]


    def _makePredictionModel(self):
        initial_state = tf.keras.layers.Input(shape=(2*self.dof,))
        final_state = tf.keras.layers.Input(shape=(2*self.dof,))

        network_input = tf.keras.layers.concatenate([initial_state, final_state], axis=-1)

        feedforward_network = tf.keras.models.Sequential()
        make_layer = lambda size: tf.keras.layers.Dense(size, activation="relu")
        for _ in range(self.numLayers):
            feedforward_network.add(make_layer(self.layerSize))
        features = feedforward_network(network_input)

        costate = tf.keras.layers.Dense(2*self.dof)(features)
        time = tf.keras.layers.Dense(1, activation =lambda x: self.maxTime*tf.nn.sigmoid(x))(features)
        cost = tf.keras.layers.Dense(1, activation =lambda x: self.maxCost*tf.nn.sigmoid(x))(features)
        reachable = tf.keras.layers.Dense(1, activation = tf.nn.sigmoid)(features)

        self.predictionModel = tf.keras.models.Model(
            inputs=[initial_state, final_state],
            outputs=[costate, time, cost, reachable]
        )


    def _makeLearningModel(self):
        initial_state = tf.keras.layers.Input(shape=(2*self.dof,))
        final_state = tf.keras.layers.Input(shape=(2*self.dof,))
        
        initial_costate = tf.keras.layers.Input(shape=(2*self.dof,))
        final_time = tf.keras.layers.Input(shape=(1,))
        final_cost = tf.keras.layers.Input(shape=(1,))

        costate, time, cost, reachable = self.predictionModel([initial_state, final_state])

        costate_error = tf.keras.losses.MeanSquaredError()(initial_costate, costate)
        time_error = tf.keras.losses.MeanSquaredError()(final_time, time)
        cost_error = tf.keras.losses.MeanSquaredError()(final_cost, cost)

        random_initial_state = tf.keras.layers.Lambda(lambda x:
                tf.constant(self.maxStates)*
                tf.random.uniform(shape=tf.shape(x),
                    minval=-1.0, maxval=1.0))(initial_state)
        random_final_state = tf.keras.layers.Lambda(lambda x:
            tf.constant(self.maxStates)*tf.random.uniform(shape=tf.shape(x),
                minval=-1.0, maxval=1.0))(final_state)

        reachable_true_target = tf.keras.layers.Lambda(lambda x:
                tf.ones(shape=tf.shape(x)))(final_time)
        reachable_false_target = tf.keras.layers.Lambda(lambda x:
                tf.zeros(shape=tf.shape(x)))(final_time)


        _, _, _, random_reachable = self.predictionModel([random_initial_state, random_final_state])

        reachable_error_true = tf.keras.losses.MeanSquaredLogarithmicError()(reachable_true_target,reachable)
        reachable_error_false = tf.keras.losses.MeanSquaredLogarithmicError()(reachable_false_target,random_reachable)


        self.learningModel = tf.keras.models.Model(
            inputs=[initial_state, final_state, initial_costate, final_time, final_cost],
            outputs=[costate, time, cost, reachable, reachable_true_target, reachable_false_target]
        )

        self.learningModel.add_loss(costate_error)
        self.learningModel.add_loss(time_error)
        self.learningModel.add_loss(cost_error)
        self.learningModel.add_loss(reachable_error_true)
        self.learningModel.add_loss(reachable_error_false)
        self.learningModel.compile(optimizer="adam")


    def loadData(self, f):
       data_lumped = np.loadtxt(f, skiprows=1, delimiter=',') 
       initial_state = data_lumped[:, :2*self.dof]
       initial_costate = data_lumped[:, 2*self.dof:4*self.dof]
       final_state = data_lumped[:, 4*self.dof:6*self.dof]
       final_cost = data_lumped[:, -2]
       final_time = data_lumped[:, -1]
       data = [initial_state, final_state, initial_costate, final_time, final_cost]
       return data 


def mainFakeData():
    datasize = 20000

    model = NonGenerativeModel()
    print(model.learningModel.weights)

    initial_state = np.random.uniform(low = -1.0, high=1.0,
            size=(datasize, 2*model.dof))
    final_state = initial_state + np.random.uniform(low = -0.05, high=-0.05,
            size=(datasize, 2*model.dof))
    initial_costate = initial_state + 0.05
    final_time = initial_state[:,0]**2
    final_cost = initial_state[:,1]**2


    model.learningModel.fit(x=[initial_state, final_state, initial_costate, final_time, final_cost], epochs = 20)
    costate, time, cost, reachable =   model.predictionModel.predict(x=[initial_state, final_state])

    final_state_fake = np.random.uniform(low = -1.0, high=1.0, size=(datasize, 2*model.dof))
    _,_,_, reachable_fake = model.predictionModel.predict([initial_state, final_state_fake]) 

    plt.ion()
    plt.figure()
    plt.subplot(4,1,1)
    plt.plot(initial_costate[:,0], costate[:,0],  "*")
    plt.subplot(4,1,2)
    plt.plot(final_time, time,  "*")
    plt.subplot(4,1,3)
    plt.plot(final_cost, cost,  "*")
    plt.subplot(4,1,4)
    plt.plot(reachable,  "*")
    plt.plot(reachable_fake, "r*")
    plt.draw()
    input("Press any key to close the plot")


def mainRealData(filename):
    config = {"dof": 3,
            "layerSize" : 256,
            "numLayers":3,
            "maxTime" : 1.0,
            "maxCost" : 1.0,
            "maxStates" : [1.6, 0.8, 0.8, 1.0, 1.0, 1.0],
            "epochs": 64,
            "batch_size": 2048
            }

    model = NonGenerativeModel(config)
    data = model.loadData(filename)
    initial_state, final_state, initial_costate, final_time, final_cost = data

    model.train(data)


    plotdatasize = 2000
    costate, time, cost, reachable = model.predict(initial_state[:plotdatasize,:], final_state[:plotdatasize,:])

   
    # Generating some fake final state data, to show if reachable works
    final_state_fake = model.maxStates*np.random.uniform(low = -1.0, high=1.0, size=(plotdatasize, 2*model.dof))
    _,_,_, reachable_fake = model.predictionModel.predict([initial_state[:plotdatasize,:], final_state_fake]) 

    plt.ion()
    plt.figure(figsize=(20,20))
    plt.subplot(4,1,1)
    plt.plot(initial_costate[:plotdatasize,0], costate[:,0],  "*")
    plt.xlabel("True initial costate")
    plt.ylabel("Estimated initial costate")

    plt.subplot(4,1,2)
    plt.plot(final_time[:plotdatasize], time,  "*")
    plt.xlabel("True final time")
    plt.ylabel("Estimated final time")

    plt.subplot(4,1,3)
    plt.plot(final_cost[:plotdatasize], cost,  "*")
    plt.xlabel("True final cost")
    plt.ylabel("Estimated final cost")

    plt.subplot(4,1,4)
    plt.plot(reachable,  "*")
    plt.plot(reachable_fake, "r*")

    plt.legend(["in data set", "fake data"])
    plt.legend("Probability point in dataset")
    plt.draw()
    plt.savefig("nongenerative_result.png")






if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="""Running non-generative NN
    colearn. Note that the file is a csv file with datapoints [initial_state,
    initial_costate, final_state, final_cost, final_time]. Fake data is used
    when no data file is given (using -f <datafile>""")

    parser.add_argument("-f", "--file", default="no_file", help="the directory in which the experiment-files can be found")
    args = parser.parse_args()

    if args.file == "no_file":
        mainFakeData()
    else:
        mainRealData(args.file)







