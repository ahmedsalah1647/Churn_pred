#
# This is a Shiny web application. You can run the application by clicking
# the 'Run App' button above.
#
# Find out more about building applications with Shiny here:
#
#    http://shiny.rstudio.com/
#

library(shinydashboard)
library(caTools)
library(e1071)
library(class)
library(caret)
# Define UI for application that draws a histogram
ui <- dashboardPage(
  dashboardHeader(title = "Options"),
  dashboardSidebar(
    sidebarMenu(
      menuItem("Preprocessing",tabName = "preprocess",icon = icon("dashboard")),
      menuItem("Train",tabName = "train", icon = icon("th")),
      menuItem("Predict",tabName = "predict",icon = icon("calendar"))
    )
  ),
  dashboardBody(
    tabItems(
      tabItem(tabName = "preprocess",
        fluidRow(
          box(
            width = 12,
            title = "View Dataset",
            status = "success",
            solidHeader = TRUE,
            dataTableOutput("df")
          ),
          actionButton( "select","Feature Selection"),
          actionButton( "handlena","Missing values treatment"),
          actionButton( "encod","Ecoding"),
          actionButton( "standard","Standardization")
        )
      ),
      tabItem(
        tabName = "train",
        selectInput('alogrithm','Choose alogrithm',choices = c("KNN"=0,"Logistic regression"=
                            1,"Naive Bayes" =2,'Decision tree'=3,'Random forest'=4,
                             "SVM"=5 )),
        actionButton('pre','Process Data'),
        actionButton('train','Train'),
        box(
          width = 12,
          title = "Confusion Matrix",
          status = "success",
          solidHeader = TRUE,
          verbatimTextOutput("cm")
        ),
        box(
          width = 12,
          title = "Accuracy",
          status = "success",
          solidHeader = TRUE,
          textOutput("mis")
        )
        ),
      tabItem(
        tabName = "predict",
        selectInput("senior","SeniorCitizian",choices = c("0"=0,"1"=1)),
        selectInput("depend","Dependents",choices = c('No'=0,'Yes'=1)),
        selectInput("secure","OnlineSecurity",choices = c("No"=0,"Yes"=1,
                                                          "No internet service"=2)),
        selectInput("tech","TechSupport",choices = c("No"=0,"Yes"=1,
                                                     "No internet service"=2)),
        selectInput("contract","Contract",choices = c("Month-to-month"=0,"Two year"=1,
                                                      "One year"=2)),
        textInput("tenure","tenure"),
        textInput("month","MonthlyCharges"),
        textInput("total","TotalCharges"),
        actionButton('predict','Predict'),
        box(
          width = 12,
          title = "Churn",
          status = "success",
          solidHeader = TRUE,
          textOutput("churn")
        )
      )
    )
  )
)

# Define server logic required to draw a histogram
server <- function(input, output) {
  df = read.csv("dataset.csv")
  assign('df',df,envir = globalenv())
  View_data(input,output)
  observeEvent(input$select,{select(input,output)})
  observeEvent(input$handlena,{nan_handle(input,output)})
  observeEvent(input$encod,{encod_data(input,output)})
  observeEvent(input$standard,{standard_data(input,output)})
  observeEvent(input$pre,{process_data(input,output)})
  observeEvent(input$train,{train_data(input,output)})
  observeEvent(input$predict,{predict_data(input,output)})
}

###############################################################################
select_feature <- function(input,output)
{
  df = df[,c('SeniorCitizen','Dependents','tenure','OnlineSecurity','TechSupport'
             ,'Contract','MonthlyCharges','TotalCharges','Churn')]
  assign('df',df,envir = globalenv())
}
handle_na <- function(input,output)
{
  t_mean = mean(df$tenure,na.rm = TRUE)
  df[is.na(df[,"tenure"]), "tenure"] <- t_mean
  df = df[!(is.na(df$SeniorCitizen)),]
  assign('df',df,envir = globalenv())
}
encod <- function(input,output)
{
  df$Dependents = factor(df$Dependents,levels = c("No","Yes"),labels = c(0,1))
  df$OnlineSecurity = factor(df$OnlineSecurity,levels = c("No","Yes",
                                                          "No internet service"),labels = c(0,1,2))
  df$TechSupport = factor(df$TechSupport,levels = c("No","Yes",
                                                    "No internet service"),labels = c(0,1,2))
  df$Contract = factor(df$Contract,levels = c("Month-to-month","Two year",
                                              "One year"),labels = c(0,1,2))
  df$Churn = factor(df$Churn,levels = c("No","Yes"),labels = c(0,1))
  assign('df',df,envir = globalenv())
}
standard <- function(input,output)
{
  t_mean = mean(df$tenure,na.rm = TRUE)
  mc_mean = mean(df$MonthlyCharges)
  tc_mean = mean(df$TotalCharges)
  t_sd = sd(df$tenure,na.rm = TRUE)
  mc_sd = sd(df$MonthlyCharges)
  tc_sd = sd(df$TotalCharges)
  assign("t_mean",t_mean,envir = globalenv())
  assign("mc_mean",mc_mean,envir = globalenv())
  assign("tc_mean",tc_mean,envir = globalenv())
  assign("t_sd",t_sd,envir = globalenv())
  assign("mc_sd",mc_sd,envir = globalenv())
  assign("tc_sd",tc_sd,envir = globalenv())
  df$tenure = (df$tenure -t_mean) / t_sd
  df$MonthlyCharges = (df$MonthlyCharges -mc_mean) / mc_sd
  df$TotalCharges = (df$TotalCharges -tc_mean) / tc_sd
  assign('df',df,envir = globalenv())
}
View_data <- function(input,output)
{
  output$df <- renderDataTable({
    
    return(df)
  },options = list(scrollX = TRUE))
  
}
select <- function(input,output)
{
  select_feature(input,output)
  View_data(input,output)
  
}
nan_handle <- function(input,output)
{
  handle_na(input,output)
  View_data(input,output)
  
}
encod_data <- function(input,output)
{
  encod(input,output)
  View_data(input,output)
  
}
standard_data <- function(input,output)
{
  standard(input,output)
  View_data(input,output)
}
split_data <- function(input,output)
{
  set.seed(123)
  split = createDataPartition(y= df$Churn, p=0.8, list = FALSE)
  training_set = df[split,]
  test_set = df[-split,]
  assign('training_set', training_set ,envir = globalenv())
  assign('test_set', test_set, envir = globalenv())
}
save_data <- function(classifier,acc,y1,y2)
{
  assign("classifier",classifier,envir = globalenv())
  assign("y",y1,envir = globalenv())
  assign("y_pred",y2,envir = globalenv())
  assign("acc",acc,envir = globalenv())
}
svm <- function()
{
  classifier = e1071::svm(formula = Churn ~ .,
                   data = training_set,
                   type = 'C-classification',
                   kernel = 'radial')
  y_pred = predict(classifier, newdata = test_set[-9])
  cm = table(test_set[, 9], y_pred)
  misClasificError <- mean(y_pred != test_set[, 9])
  acc <- (1-misClasificError)*100
  save_data(classifier,acc,test_set[, 9],y_pred)
}
decision_tree <- function()
{
  require(tree)
  tree_model <- tree(Churn ~ ., training_set)
  y_pred <- predict(tree_model,newdata = test_set[-9],type = 'class')
  misClasificError <- mean(y_pred != test_set[, 9])
  acc <- (1-misClasificError)*100
  save_data(tree_model,acc,test_set[, 9],y_pred)
}
log_reg <- function(input,output)
{
  log_model <- glm(formula = Churn ~ . ,family = binomial,
                   data = training_set)
  
  prob_pred = predict(log_model, type = 'response', newdata = test_set[-9])
  y_pred = ifelse(prob_pred > 0.4, 1, 0)
  misClasificError <- mean(y_pred != test_set[, 9])
  acc <- (1-misClasificError)*100
  save_data(log_model,acc,test_set[, 9],y_pred)
}
knn <- function(input,output)
{
  knn_model <- class::knn(train = training_set[-9],test = test_set[-9],cl = training_set[,9],k = 7)
  misClasificError <- mean(knn_model != test_set[,9])
  acc <- (1-misClasificError)*100
  save_data(knn_model,acc,test_set[, 9],knn_model)
}
random_forest <-function(input,output)
{
  require(randomForest)
  forest_model <- randomForest(Churn ~ ., training_set, ntree = 600, mtry = 4, importance = TRUE)
  forest_res <- predict(forest_model,newdata = test_set[-9],type = 'class')
  misClasificError <- mean(forest_res != test_set[, 9])
  acc <- (1-misClasificError)*100
  save_data(forest_model,acc,test_set[, 9],y_pred)
}
naive_Bayes<- function(input,output)
{
  classifier = naiveBayes(x = training_set[-9],
                          y = training_set$Churn)
  y_pred = predict(classifier, newdata = test_set[-9])
  misClasificError <- mean(y_pred != test_set[, 9])
  acc <- (1-misClasificError)*100
  save_data(classifier,acc,test_set[, 9],y_pred)
}
process_data <- function(input,output)
{
  select_feature(input,output)
  handle_na(input,output)
  encod(input,output)
  standard(input,output)
  View_data(input,output)
}
train_data<- function(input,output)
{
  split_data(input,output)
  a <- input$alogrithm
  assign("a",a,envir = globalenv())
  if (a==5)
  svm()
  if (a==3)
  decision_tree()
  if(a==1)
  log_reg(input,output)
  if (a==0)
  knn(input,output)
  if (a==4)
  random_forest(input,output)
  if(a==2)
  naive_Bayes(input,output)
  output$cm <- renderPrint({
    
    table(y_pred,y)
  })
  output$mis <- renderText({
    
    return(acc)
  })
}
predict_data <- function(input,output)
{
  t = as.double(input$tenure)
  mc = as.double(input$month)
  tc = as.double(input$total)
  t = (t -t_mean) / t_sd
  mc = (mc -mc_mean) / mc_sd
  tc = (tc -tc_mean) / tc_sd
  x = c(input$senior,input$depend,t,input$secure,input$tech,
        input$contract,mc,tc)
  if (a==0)
    y_pred = class::knn(train = training_set[-9],test = x,cl = training_set[,9],k = 7)
  else
    y_pred = predict(classifier, newdata = test_set[-9])
  if (a==1)
    y_pred = ifelse(y_pred > 0.4, 1, 0)
  output$churn <- renderText({
    if (y_pred==0)
     return("Yes")
    else
      return("No")
  })

}
# Run the application 
shinyApp(ui = ui, server = server)

