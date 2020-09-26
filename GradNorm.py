import tensorflow.compat.v1 as tf

import main


class GradNorm():
    def __init__(self, statisticsPrefix, useAdaptiveLossBalancing, adaptiveGradBalancingRatio):

        self.dLossGradMA = 1.0
        self.statisticsPrefix = statisticsPrefix

        self.useAdaptiveLossBalancing = useAdaptiveLossBalancing
        self.adaptiveGradBalancingRatio = adaptiveGradBalancingRatio
        self.beta = 10.0
        self.maGAN = 1.0
        self.maMSE = 1.0
        self.maGAN_debug = 1.0
        self.count_train_steps = 0

        self.rate = 0.01

    def handleNormsAGB(self, sess, feed_dict):

        feed_dict.update({"betaPL:0": self.beta})

        gradGANMeanVal, gradMSEMeanVal = sess.run(["gradPredGANMeanHook:0", "gradPredMSEMeanHook:0"],
                                                  feed_dict=feed_dict)

        self.maGAN = self.maGAN * (1.0 - self.rate) + self.rate * gradGANMeanVal
        self.maMSE = self.maMSE * (1.0 - self.rate) + self.rate * gradMSEMeanVal
        self.maGAN_debug = self.maGAN_debug * (1.0 - self.rate) + self.rate * gradGANMeanVal

        if self.count_train_steps == 0:
            self.maGAN = gradGANMeanVal
            self.maMSE = gradMSEMeanVal
            self.writer = tf.summary.FileWriter("./" + main.RUN_NAME, sess.graph)

        self.count_train_steps += 1
        if (self.count_train_steps % 100 == 0):
            debug_summary = tf.Summary()
            debug_summary.value.add(tag="maMSE", simple_value=self.maMSE)
            debug_summary.value.add(tag="maGAN", simple_value=self.maGAN)
            debug_summary.value.add(tag="maGAN_debug", simple_value=self.maGAN_debug)
            self.writer.add_summary(debug_summary, self.count_train_steps)

        if (self.useAdaptiveLossBalancing):
            if (self.maGAN > self.maMSE * self.adaptiveGradBalancingRatio and self.count_train_steps > 1):
                self.beta = self.beta * (1.0 + self.rate)
                self.maGAN = self.maGAN * (1.0 - self.rate)
                print("self.maMSE: " + str(self.maMSE) + " self.maMSE*adaptiveGradBalancingRatio:" + str(
                    self.maMSE * self.adaptiveGradBalancingRatio) + " self.reconGanModel.gradBalance: " + str(
                    self.beta))

        return feed_dict
