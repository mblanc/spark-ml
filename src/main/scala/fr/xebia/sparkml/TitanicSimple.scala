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

object TitanicSimple {

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

    // spark-csv assumes each column is of type string
    // select only the useful columns, rename them and cast them to the right type
    val df = csv.select(
      $"Survived".as("label").cast(DoubleType),
      $"Age".as("age").cast(IntegerType),
      $"Fare".as("fare").cast(DoubleType),
      $"Pclass".as("class").cast(DoubleType)
    )

    // We replace the missing values of the age and fare columns by their mean.
    val set = df.na.fill(Map("age" -> 30, "fare" -> 32.2))

    // We will train our model on 75% of our data and use the 25% left for validation.
    val Array(trainSet, validationSet) = set.randomSplit(Array(0.75, 0.25))

    // The stages of our pipeline
    val classEncoder = new OneHotEncoder()
      .setInputCol("class")
      .setOutputCol("classVec")
    val vectorAssembler = new VectorAssembler()
      .setInputCols(Array("age", "fare", "classVec"))
      .setOutputCol("features")
    val logisticRegression = new LogisticRegression()
      .setMaxIter(50)
      .setRegParam(0.01)

    // the pipeline
    val pipeline = new Pipeline()
      .setStages(Array(classEncoder, vectorAssembler, logisticRegression))

    println("Train")
    val pipelineModel = pipeline.fit(trainSet)

    println("Evaluate the model on the validation set.")
    val validationPredictions = pipelineModel.transform(validationSet)

    // We evaluate the model
    // Print the area under the ROC curve for the validation set
    val binaryClassificationEvaluator: BinaryClassificationEvaluator = new BinaryClassificationEvaluator()
    println(s"${binaryClassificationEvaluator.getMetricName} ${binaryClassificationEvaluator.evaluate(validationPredictions)}")

    // Lets make prediction on new data where the label is unknown
    println("Predict test.csv passengers fate")
    val csvTest = sqlContext.read.format("com.databricks.spark.csv").option("header", "true").load("src/main/resources/titanic/test.csv")

    val dfTest = csvTest.select(
      $"PassengerId",
      $"Age".as("age").cast(IntegerType),
      $"Fare".as("fare").cast(DoubleType),
      $"Pclass".as("class").cast(DoubleType),
      $"Name".as("name"),
      $"Sex".as("sex")
    ).coalesce(1)

    val selectTest = dfTest.na.fill(Map("age" -> 30, "fare" -> 32.2))

    // We make predictions on the test dataset.
    val result = pipelineModel.transform(selectTest)

    result.show()

    // let's write the result in the correct format for Kaggle
    result.select($"PassengerId", $"prediction".cast(IntegerType))
      .write.format("com.databricks.spark.csv").save("src/main/resources/titanic/result")
  }


}
