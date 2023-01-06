#'@title Read GUD_OPTIONS.h input file for MITgcm GUD module.
#'@description
#'Read GUD_OPTIONS.h input file and create a list of keys (code options).
#'
#'@param none for now
#'@return opt list
#'@author
#'F. Maps 2020
#'@export

require("dplyr")

read.options <-	function() {

  # Read GUD_OPTIONS.h input file with def keys
  infile <- "../../../gud_1d_35+16/code_noradtrans/GUD_OPTIONS.h"
  if( !file.exists( infile ) ) {
    infile <- "../../../pkg/gud/GUD_OPTIONS.h"
  }
  df      <- file( infile, "r" ) # read only
  opt_txt <- readLines( df, skipNul = TRUE )
  close( df )

  # Select only relevant lines
  keep  <- grepl( "define|undef", opt_txt )
  opt_txt <- opt_txt[ keep ]

  # Create the list of GUD options
  opt <- list()

  # FALSE keys
               undef  <- grepl( "undef",  opt_txt )
         opt  [undef] <- FALSE
  names( opt )[undef] <- strsplit( opt_txt[undef], "#undef " ) %>% # remove keyword
                         sapply( "[", 2 )                      %>% # get its name
                         trimws( which = "both" )                  # get rid of useless white spaces

  # TRUE keys
               define  <- grepl( "define",  opt_txt )
         opt  [define] <- TRUE
  names( opt )[define] <- strsplit( opt_txt[define], "#define " ) %>%
                          sapply( "[", 2 )                        %>%
                          trimws( which = "both" )

  # Numerical keys
  num_key <- grep( " ", names(opt) )
  for ( i in 1:length(num_key) ) {
           opt  [ num_key[i] ] <- strsplit( names( opt[ num_key[i] ] ), " " ) %>% 
                                  sapply( "[", 2 )                            %>% 
                                  as.numeric()
    names( opt )[ num_key[i] ] <- strsplit( names( opt[ num_key[i] ] ), " " ) %>% 
                                  sapply( "[", 1 )                            %>%
                                  trimws( which = "both" )
  }

  # Return the list of GUD option keys
  return(opt)
}
