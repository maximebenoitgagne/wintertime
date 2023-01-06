#'@title Read in data.* namelists input files from MITgcm GUD module.
#'@description
#'Read data.* input files and create a list from namelist sections.
#'
#'@param params a list of parameters dimensions and default values to be updated
#'@return params list with updated default parameters
#'@author
#'F. Maps 2020
#'@export


require("dplyr")

read.nml <-	function( params ) {

  
  #--- Read namelists in data.* input files

  infile1 <- "../../../gud_1d_35+16/input_noradtrans/data.gud"
  infile2 <- "../../../gud_1d_35+16/input_noradtrans/data.traits"
  df1     <- file( infile1, "r" ); df2 <- file( infile2, "r" )
  nml_txt <- c( readLines( df1, skipNul = TRUE ),
                readLines( df2, skipNul = TRUE ) ) %>%
             trimws( which = "both" ) # Trim white spaces (1/2)
  close( df1 ); close( df2 )

  # Ignore comment/empty lines
  nml_1    <- substr( nml_txt, 1, 1 )
  skip  	 <- nml_1 == "C" |
              nml_1 == "!" |
              nml_1 == "#" |
              nml_1 == "&" |
              nml_1 == "/" |
              nml_1 == ""
  
  nml_txt  <- nml_txt[!skip]                                    %>% # Ignore comment lines
              strsplit( split = "!" ) %>% sapply( "[", 1 )      %>% # Remove trailing comments
              trimws( which = "both" )                              # Trim white spaces (2/2)

  par_name <- NULL
  par_val  <- NULL
  
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
  
  l <- length( params )
  for ( i in 1:length(nml_txt) ) {

    par_oldName <- par_name
    par_oldVal  <- par_val
    
    nml_tmp     <- nml_txt[i]

    # Check that a value is assigned to the parameter; otherwise ignore
    if( grepl( "=", nml_tmp ) ) {
      
      par_name <- strsplit( nml_tmp, "=" )[[1]][1] %>% trimws( which = "both" )
      par_val  <- strsplit( nml_tmp, "=" )[[1]][2]
      par_val  <- substr( par_val, 1, nchar(par_val)-1 ) # Get rid of trailing comma (THERE SHOULD BE ONE!)

    } else {
      warning( paste( "!!! There is no value assigned to parameter", par_name, "inside namelist", nml_name, "!!!" ) )
      next
    }
    
    # Deal with scalar parameters first
    if( !grepl( "[(]", par_name ) ) {
      
      # Check whether the value is numeric...
      par_tmp <- as.numeric( par_val )
      if( is.finite(par_tmp) ) {
        par_val <- par_tmp
      #... or character...
      } else if( grepl( "'", par_val ) ) {
        par_val <- gsub( "'", "", par_val )
      } else if( grepl( "\"", par_val ) ) {
        par_val <- gsub( "\"", "", par_val )
      #... or logic
      } else if( grepl( "T", par_val ) ) {
        par_val <- TRUE
      } else if( grepl( "F", par_val ) ) {
        par_val <- FALSE
      }
      
      # Check whether parameter has already been declared with a default value
      k <- grep( paste0('^',par_name,'$'), names(params), ignore.case = TRUE )
      if( length( k ) > 0 ) {
        # Update
              params [[k]] <- par_val
      } else {
        # Append to list
                       l   <- l + 1
              params [[l]] <- par_val
        names(params)[[l]] <- par_name
      }
      
    # Then deal with vectors and matrices (!!! ASSUME NON-SCALAR PARAMETER TO BE NUMERIC !!!)
    } else {
      
      par_dim  <- strsplit( strsplit( par_name, "[(]" )[[1]][2], "[)]" )[[1]][1] %>% trimws( which = "both" )
      par_name <- strsplit( par_name, "[(]" )[[1]][1]                            %>% trimws( which = "both" )
      
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
            times   <- as.numeric( strsplit( par_spt[[1]][k], "[*]" )[[1]][1] )
            val     <- as.numeric( strsplit( par_spt[[1]][k], "[*]" )[[1]][2] )
            par_val <- c( par_val, rep( val, times ) )
          } else {
            par_val <- c( par_val, as.numeric( par_spt[[1]][k] ) )
          }
        }
      }
      
      # Check special case for dimensions specification
      k <- gregexpr( ":|,", par_dim ) %>% unlist()
      if( length(k) == nchar(par_dim) ) {
        par_dim <- gsub( ":", "", par_dim )
      }

      # Check whether parameter has already been declared with default values
      k <- grep( paste0('^',par_name,'$'), names(params), ignore.case = TRUE )
      if( length(k) > 0 ) {
        # Update
        eval( parse( text = paste0( 'params[[k]][',par_dim,'] <- par_val' ) ) )
      } else {
        # Append to list
                l   <- l + 1
        params[[l]] <- NULL
        eval( parse( text = paste0( 'params[[l]][',par_dim,'] <- par_val' ) ) )
        names( params )[[l]] <- par_name
      }
    }
    
  } # end for

  # Return the values defined in the namelists
  return( params )
}
