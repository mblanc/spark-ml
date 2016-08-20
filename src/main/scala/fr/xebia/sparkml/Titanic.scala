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

object Titanic {



  def main(args: Array[String]) {
    FileUtils.deleteDirectory(new File("src/main/resources/titanic/result"))

    val conf = new SparkConf().setAppName("Titanic").setMaster("local[*]")

    val sc = new SparkContext(conf)

    val sqlContext = new SQLContext(sc)
    // We use the $ operator from implicit class StringToColumn
    import sqlContext.implicits._

    // Load the train.csv file as a DataFrame
    // Use the spark-csv library see https://github.com/databricks/spark-csv
    val csv = sqlContext.read.format("com.databricks.spark.csv").option("header", "true").load("src/main/resources/titanic/train.csv")


    // spark-csv put the type StringType to each column
    csv.printSchema()

    // select only the useful columns, rename them and cast them to the right type

    val df = selectCols(csv)

    // verify the schema
    df.printSchema()

    // look at the data
    df.show()

    // show stats for each column
    df.describe(df.columns: _*).show()


    // We replace the missing values of the age and fare columns by their mean.
    import org.apache.spark.sql.functions._
    val average_age = df.select(avg($"age"))
    val average_fare = df.select(avg($"fare"))
    val select = df.na.fill(Map("age" -> average_age, "fare" -> average_fare))

    // We will train our model on 75% of our data and use the 25% left for validation.
    val Array(trainSet, validationSet) = select.randomSplit(Array(0.75, 0.25))

    // The stages of our pipeline
    val sexIndexer = new StringIndexer().setInputCol("sex").setOutputCol("sexIndex")
    val classEncoder = new OneHotEncoder().setInputCol("class").setOutputCol("classVec")
    val tokenizer = new Tokenizer().setInputCol("name").setOutputCol("words")
    val hashingTF = new HashingTF().setInputCol(tokenizer.getOutputCol).setOutputCol("hash")
    val vectorAssembler = new VectorAssembler().setInputCols(Array("hash", "age", "fare", "sexIndex", "classVec")).setOutputCol("features")
    val logisticRegression = new LogisticRegression()
    val pipeline = new Pipeline().setStages(Array(sexIndexer, classEncoder, tokenizer, hashingTF, vectorAssembler, logisticRegression))

    // We will cross validate our pipeline
    val crossValidator = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(new BinaryClassificationEvaluator)

    // Here are the params we want to validationPredictions
    val paramGrid = new ParamGridBuilder()
      .addGrid(hashingTF.numFeatures, Array(2, 5, 1000))
      .addGrid(logisticRegression.regParam, Array(1, 0.1, 0.01))
      .addGrid(logisticRegression.maxIter, Array(10, 50, 100))
      .build()
    crossValidator.setEstimatorParamMaps(paramGrid)

    // We will use a 3-fold cross validation
    crossValidator.setNumFolds(3)

    println("Cross Validation")
    val cvModel = crossValidator.fit(trainSet)

    println("Best model")
    for (stage <- cvModel.bestModel.asInstanceOf[PipelineModel].stages) println(stage.explainParams())

    println("Evaluate the model on the validation set.")
    val validationPredictions = cvModel.transform(validationSet)

    // Area under the ROC curve for the validation set
    val binaryClassificationEvaluator: BinaryClassificationEvaluator = new BinaryClassificationEvaluator()
    println(s"${binaryClassificationEvaluator.getMetricName} ${binaryClassificationEvaluator.evaluate(validationPredictions)}")

    // We want to print the percentage of passengers we correctly predict on the validation set
    val total = validationPredictions.count()
    val goodPredictionCount = validationPredictions.filter(validationPredictions("label") === validationPredictions("prediction")).count()
    println(s"correct prediction percentage : ${goodPredictionCount / total.toDouble}")


    // Lets make prediction on new data where the label is unknown
    println("Predict validationPredictions.csv passengers fate")
    val csvTest = sqlContext.read.format("com.databricks.spark.csv").option("header", "true").load("src/main/resources/titanic/test.csv")

    val dfTest = csvTest.select(
      $"PassengerId",
      $"Age".as("age").cast(IntegerType),
      $"Fare".as("fare").cast(DoubleType),
      $"Pclass".as("class").cast(DoubleType),
      $"Sex".as("sex"),
      $"Name".as("name")
    ).coalesce(1)

    val selectTest = dfTest.na.fill(Map("age" -> 30, "fare" -> 32.2))

    //
    val result = cvModel.transform(selectTest)

    // let's write the result in the correct format for Kaggle
    result.select($"PassengerId", $"prediction".cast(IntegerType))
      .write.format("com.databricks.spark.csv").save("src/main/resources/titanic/result")
  }


}
