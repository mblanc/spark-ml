name := "spark-ml"

version := "1.0"

scalaVersion := "2.10.4"

resolvers += Resolver.mavenLocal

val sparkVersion: String = "1.4.0"

libraryDependencies += "org.apache.spark" %% "spark-core" % sparkVersion withSources() withJavadoc()

libraryDependencies += "org.apache.spark" %% "spark-mllib" % sparkVersion withSources() withJavadoc()

libraryDependencies += "org.apache.spark" %% "spark-sql" % sparkVersion withSources() withJavadoc()

libraryDependencies += "com.databricks" %% "spark-csv" % "1.1.0" withSources() withJavadoc()
    