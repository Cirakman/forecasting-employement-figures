

library(forecast)

runtbats <- function(y, h) {
 	fit <- tbats(y)
	return(forecast(fit,h))
}
