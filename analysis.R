
### Company Salary Analysis
library(dplyr)
company <- salary %>% 
  group_by(Company) %>%
  summarise(n = n(),
            min = min(SalaryNormalized),
            median = median(as.numeric(SalaryNormalized)),
            mean = mean(SalaryNormalized),
            max = max(SalaryNormalized))
company <- arrange(company, desc(n))

### Title Salary Analysis
title.senior <- grepl("senior", title.v)
sum(title.senior)
mean(salary$SalaryNormalized[title.senior])
title.manager <- grepl("manag", title.v)
sum(title.manager)
mean(salary$SalaryNormalized[title.manager])
title.lead <- grepl("lead", title.v)
sum(title.lead)
mean(salary$SalaryNormalized[title.lead])

###
grepl("Java", salary$Title[1:10000]) & grepl("London", salary$LocationRaw[1:10000])
salary[9393,]
salary[9753,]
salary[9063,]
apply(salary, 2, function(x){
  sum(is.na(x))
})

length(unique(salary[[2]]))
for(i in c(2,4,5,8)){
  print(length(unique(salary[[i]])))
}
print(length(unique(salary[[14]])))

###
table(salary$Category)
