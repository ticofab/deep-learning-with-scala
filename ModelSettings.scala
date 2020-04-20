package io.ticofab.piai.learning

import java.io.File
import java.util.Random

import org.datavec.api.io.labels.ParentPathLabelGenerator
import org.datavec.api.split.{FileSplit, InputSplit}
import org.datavec.image.loader.BaseImageLoader
import org.datavec.image.recordreader.ImageRecordReader
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.conf.layers._
import org.deeplearning4j.nn.conf.{MultiLayerConfiguration, NeuralNetConfiguration}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.learning.config.Adam
import org.nd4j.linalg.lossfunctions.LossFunctions

object Settings {

  // information of the non-flattened images
  val outputHeight = 25 // pixels
  val outputWidth = 171 // pixels
  val depth = 3 // Number of input channels, or depth - 3 because we're in RGB setting

  // training settings
  val numClasses = 2 // The number of possible outcomes
  val batchSize = 50 // Test batch size
  val nEpochs = 4 // Number of training epochs
  val trainSetPercentage = 80.01

  // data settings
  val seed = 456 // the random seed
  val rng = new Random(seed)
  val rootDir: File = FILE_POINTING_TO_THE_IMAGES_ROOT_DIRECTORY
  val allowedExtensions: Array[String] = BaseImageLoader.ALLOWED_FORMATS
  val labelMaker = new ParentPathLabelGenerator()
  val fileSplit = new FileSplit(rootDir, allowedExtensions, rng)

  // evaluation settings
  val scoreIterationListenerPrintIterations = 50
  val evalutativeListenerFrequency = 2
  val saveModel = true

  // creates the actual neural network
  def getNetworkConfiguration: MultiLayerConfiguration = {
    new NeuralNetConfiguration.Builder()
      .seed(seed)
      .l2(0.0005)
      .weightInit(WeightInit.XAVIER)
      .updater(new Adam(1e-3))
      .list
      .layer(0, new ConvolutionLayer.Builder(5, 5)
        .nIn(depth) // nIn and nOut specify depth. nIn here is the nChannels and nOut is the number of filters to be applied
        .stride(1, 1)
        .nOut(20)
        .activation(Activation.IDENTITY)
        .build)
      .layer(1, new SubsamplingLayer.Builder(PoolingType.MAX)
        .kernelSize(2, 2)
        .stride(2, 2)
        .build)
      .layer(2, new ConvolutionLayer.Builder(5, 5)
        .stride(1, 1) // Note that nIn need not be specified in later layers
        .nOut(50)
        .activation(Activation.IDENTITY)
        .build)
      .layer(3, new SubsamplingLayer.Builder(PoolingType.MAX)
        .kernelSize(2, 2)
        .stride(2, 2)
        .build)
      .layer(4, new DenseLayer.Builder()
        .activation(Activation.RELU)
        .nOut(500)
        .build)
      .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT) // to be used with Softmax
        .nOut(numClasses)
        .activation(Activation.SOFTMAX) // because I have two classes: plane or no-plane
        .build)
      .setInputType(InputType.convolutional(outputHeight, outputWidth, depth))
      .build
  }
  
  // creates an image scanner with the accurate settings
  def getInitializedIterator(inputSplit: InputSplit): RecordReaderDataSetIterator = {
    val ir = new ImageRecordReader(outputHeight, outputWidth, depth, labelMaker)
    ir.initialize(inputSplit)
    new RecordReaderDataSetIterator(ir, batchSize, 1, numClasses) // always 1 for image record reader
  }
}
