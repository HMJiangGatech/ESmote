library(esmote)
context("Test Utils")

test_that('distance is Euclidean distance',
          {
            expect_equal(distance(c(4.1,-1.5,6.3), c(5.2,0.7,-2)),
                         sum(c(1.21,4.84,68.89))^0.5)
            expect_equal(distance(c(5.2,0.7,-2), c(4.1,-1.5,6.3)),
                         sum(c(1.21,4.84,68.89))^0.5)
          })
