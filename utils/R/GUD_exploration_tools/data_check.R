#'@title Read in gud_traits.txt output file from MITgcm GUD module.
#'@description
#'Read gud_traits.txt to double check the values obtained from namelists & *.F functions.
#'
#'@param params a list of parameters dimensions and default values to be updated
#'@return A list of problematic values, if any
#'@author
#'F. Maps 2020
#'@export


require("dplyr")

check.pars <-	function( params ) {

  
  #--- Read gud_traits.txt output files

  infile  <- "../../../gud_1d_35+16/gud_traits.txt"
  df      <- file( infile, "r" )
  gud_txt <- c( readLines( df, skipNul = TRUE ) ) %>%
             trimws( which = "both" ) # Trim white spaces (1/2)
  close( df )

  # Ignore comment/empty lines
  gud_1    <- substr( gud_txt, 1, 1 )
  skip  	 <- gud_1 == "&" |
              gud_1 == "/"
  
  gud_txt  <- gud_txt[!skip]                                %>% # Ignore comment lines
              strsplit( split = "!" ) %>% sapply( "[", 1 )  %>% # Remove trailing comments
              trimws( which = "both" )                          # Trim white spaces (2/2)

  par_name <- NULL
  par_val  <- NULL; par_oldVal <- par_val
  
  # In the following, explore sequentially the namelist string as follows:
  #     4    | 2 | 6 | 5 | 6 |   | 1 |    3
  # par_name | ( | i | , | j | ) | = | par_value
  # 1: check first that a value is assigned to a parameter; otherwise skip
  # 2: check whether the parameter is a scalar
  # 3: if so, *) explore the content of the value string to detect its mode ( numeric | logical | character )
  # 4:        *) check if the parameter has already been defined with a default value; update value if needed
  #    if not, continue exploring the parameter's dimension(s)
  #!!! A NON-SCALAR PARAMETER IS ASSUMED TO BE NUMERIC !!!
  # 5: check if the parameter is a matrix (i,j) or a vector (i)
  # 4: check (again) if the parameter has already been defined with a default value; update value(s) if needed
  # 6: check if the assignment is for the whole [vector|array], or for specific (integer) coordinates
  
  checl <- list()
  l     <- 0

  for ( i in 1:length(gud_txt) ) {
  
    gud_tmp     <- gud_txt[i]

    # Check that this line assigns a value to a parameter
    if( grepl( "=", gud_tmp ) ) {
      
      par_name <- strsplit( gud_tmp, "=" )[[1]][1] %>% trimws( which = "both" )
      par_val  <- strsplit( gud_tmp, "=" )[[1]][2]
      par_val  <- substr( par_val, 1, nchar(par_val)-1 ) %>% trimws( which = "both" ) # Get rid of trailing comma (THERE SHOULD BE ONE!)

      # Build the vector of values according to FORTRAN formats
      if( !grepl( "[*]", par_val ) ) {
        # Simple comma-separated vector of numeric values...
        par_val <- as.numeric( unlist( strsplit( par_val, "," ) ) )
        #... or complex FORTRAN format
      } else {
        par_spt <- strsplit( par_val, "," )
        par_val <- NULL
        for( k in 1:length(par_spt[[1]]) ) {
          if( grepl( "[*]", par_spt[[1]][k] ) ) {
            dims    <- strsplit( par_spt[[1]][k], "[*]" )[[1]]
            times   <- as.numeric( dims[1] )
            val     <- as.numeric( dims[2] )
            par_val <- c( par_val, rep( val, times ) )
          } else {
            par_val <- c( par_val, as.numeric( par_spt[[1]][k] ) )
          }
        }
      }

      par_oldVal <- par_val

      # Append to list
                    l   <- l + 1
            checl [[l]] <- par_oldVal
      names(checl)[[l]] <- par_name
      
    } else {

      par_val  <- substr( gud_tmp, 1, nchar(gud_tmp)-1 ) %>% trimws( which = "both" ) # Get rid of trailing comma (THERE SHOULD BE ONE!)

      # Build the vector of values according to FORTRAN formats
      if( !grepl( "[*]", par_val ) ) {
        # Simple comma-separated vector of numeric values...
        par_val <- as.numeric( unlist( strsplit( par_val, "," ) ) )
        #... or complex FORTRAN format
      } else {
        par_spt <- strsplit( par_val, "," )
        par_val <- NULL
        for( k in 1:length(par_spt[[1]]) ) {
          if( grepl( "[*]", par_spt[[1]][k] ) ) {
            dims    <- strsplit( par_spt[[1]][k], "[*]" )[[1]]
            times   <- as.numeric( dims[1] )
            val     <- as.numeric( dims[2] )
            par_val <- c( par_val, rep( val, times ) )
          } else {
            par_val <- c( par_val, as.numeric( par_spt[[1]][k] ) )
          }
        }
      }

      par_oldVal  <- c( par_oldVal, par_val )
      
      # Update
      checl[[l]] <- par_oldVal
    }
    
  } # end for

  # Return the values computed by GUD
  return( checl )
}
