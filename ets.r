

library(forecast)

runets <- function(y, h) {
 	fit <- ets(y)
	return(forecast(fit,h))
}
