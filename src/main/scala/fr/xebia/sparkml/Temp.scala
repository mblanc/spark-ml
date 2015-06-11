package fr.xebia.sparkml

import java.io.File

import org.apache.commons.io.FileUtils
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature._
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.types._
import org.apache.spark.{SparkConf, SparkContext}

object Temp {

  def main(args: Array[String]) {
    FileUtils.deleteDirectory(new File("src/main/resources/titanic/result"))

    val conf = new SparkConf()
      .setAppName("Titanic")
      .setMaster("local[*]")

    val sc = new SparkContext(conf)

    val sqlContext = new SQLContext(sc)
    // We use the $ operator from implicit class StringToColumn
    import sqlContext.implicits._

    // Load the train.csv file as a DataFrame
    // Use the spark-csv library see https://github.com/databricks/spark-csv
    val csv = sqlContext.read.format("com.databricks.spark.csv").option("header", "true").load("src/main/resources/titanic/train.csv")
    val trainDF = sqlContext.read.format("com.databricks.spark.csv").option("header", "true").load("src/main/resources/titanic/train.csv")
    val testDF = sqlContext.read.format("com.databricks.spark.csv").option("header", "true").load("src/main/resources/titanic/train.csv")

    // spark-csv put the type StringType to each column
    csv.printSchema()

    // select only the useful columns, rename them and cast them to the right type
    var df = csv.select(
      $"Survived".as("label").cast(DoubleType),
      $"Age".as("age").cast(IntegerType),
      $"Fare".as("fare").cast(DoubleType),
      $"Pclass".as("class").cast(DoubleType),
      $"Sex".as("sex"),
      $"Name".as("name")
    )

    // verify the schema
    df.printSchema()

    // look at the data
    df.show()

    // show stats for each column
    df.describe(df.columns: _*).show()

    // StringIndexer is a Model.
    // Models are Transformers but cannot be use in a pipeline.
    // Will be fixed in 1.4.1 (see SPARK-8151)
    val sexIndexer = new StringIndexer()
      .setInputCol("sex")
      .setOutputCol("sexIndex")
      .fit(df)
    df = sexIndexer.transform(df)
    println("sexIndexer")
    df.printSchema()

    // We replace the missing values of the age and fare columns by their mean.
    val dataFrame = df.na.fill(Map("age" -> 30, "fare" -> 32.2))

    // The stages of our pipeline

    // Add a categoryVec to the DataFrame by applying OneHotEncoder transformation to the column category
    val classEncoder = new OneHotEncoder()
      .setInputCol("catergory")
      .setOutputCol("catergoryVec")

    val newDataFrame = classEncoder.transform(dataFrame)

    println("classEncoder OneHotEncoder")
    df.printSchema()

    val tokenizer = new Tokenizer().setInputCol("name").setOutputCol("words")

    df = tokenizer.transform(df)
    println("tokenizer Tokenizer")
    df.printSchema()
    df.take(5).foreach(println)


    val hashingTF = new HashingTF().setNumFeatures(5).setInputCol(tokenizer.getOutputCol).setOutputCol("hash")

    df = hashingTF.transform(df)
    println("hashingTF HashingTF")
    df.take(5).foreach(println)
    df.show()

    val vectorAssembler = new VectorAssembler().setInputCols(Array("hash", "age", "fare", "sexIndex", "classVec")).setOutputCol("features")

    df = vectorAssembler.transform(df)
    println("vectorAssembler VectorAssembler")
    df.printSchema()

    // Apply logisticRegression on a training dataset to create a model
    // used to compute predictions on a test dataset
    val logisticRegression = new LogisticRegression()
      .setMaxIter(50)
      .setRegParam(0.01)

    val lrModel = logisticRegression.fit(trainDF)

    val dataFrameWithLabelAndPrediction = lrModel.transform(testDF)

    dataFrameWithLabelAndPrediction.show()

    // Area under the ROC curve for the validation set
    val evaluator = new BinaryClassificationEvaluator()
    println(evaluator.evaluate(dataFrameWithLabelAndPrediction))

  }


}
