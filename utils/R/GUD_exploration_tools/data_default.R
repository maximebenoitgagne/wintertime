#'@title Read default values of parameters from MITgcm GUD module.
#'@description
#'Read FORTRAN header and code files and create a list of parameters:
#'GUD_SIZE.h, GUD_TRAITS.h, GUDTRAITPARAMS.h
#'gud_readtraitparams.F
#'
#'@param nothing yet
#'@return A list of variables defined BEFORE the namelists are read
#'@author
#'F. Maps 2020
#'@export

require("dplyr")

read.default <-	function( opt ) {
  
  #--- opt : GUD option keys

  #--- Read dimensions and indices ranges

  # Check whether the header file has been locally modified
  infile <- "../../../gud_1d_35+16/code_noradtrans/SIZE.h"
  if( !file.exists( infile ) ) {
    infile <- "../../../pkg/gud/SIZE.h"
  }
  df1 <- file( infile, "r" ) # read only

  infile <- "../../../gud_1d_35+16/code_noradtrans/GUD_SIZE.h"
  if( !file.exists( infile ) ) {
    infile <- "../../../pkg/gud/GUD_SIZE.h"
  }
  df2 <- file( infile, "r" ) # read only

  dim_txt <- c( readLines( df1, skipNul = T ),
                readLines( df2, skipNul = T ) )
  
  close( df1 ); close( df2 )

  # Get parameter definitions
  dim_1 <- substr( dim_txt, 1, 1 )
  skip  <- dim_1 == "C" |
           dim_1 == "c" |
           dim_1 == "#" 
  keep  <- grepl( "=", dim_txt )

  dim_txt <- dim_txt[ !skip & keep ]                                          %>% # Keep parameter declarations only
             gsub( pattern = "&|[(]|[)]", replacement = "" )                  %>% # Remove declarations
              sub( pattern = "parameter", replacement = "", ignore.case = T ) %>% 
             trimws( which = "both" )                                             # Trim white spaces

  # Create list of dimensions and ranges
  dim_list <- list()
  
  l <- 0
  for( i in 1:length(dim_txt) ) {
  l <- l + 1

    # Check whether there is only one parameter per line...
    jx <- regexpr( ",", dim_txt[i] )

    # if so...
    if( jx == -1 ) {
      
      dim_tmp <- strsplit( dim_txt[i], "=" )
      
      # Add the parameter's name & value in the list returned
      val_tmp  <-   eval( parse( text =   dim_tmp[[1]][2] ) )
      name_tmp <- trimws( which = "both", dim_tmp[[1]][1]   )

             dim_list  [[l]] <- val_tmp
      names( dim_list )[[l]] <- name_tmp
      
      # Create the variable in the working space
      assign( name_tmp, val_tmp )
      
    # if not...
    } else {
      
      # ...split the parameters on the line
      dim_spl <- strsplit( dim_txt[i], "," ) %>% unlist()
      
      for( j in 1:length(dim_spl) ) {
      l <- l + j - 1

        dim_tmp  <- strsplit( dim_spl[j], "=" ) %>% unlist()
        
        val_tmp  <-   eval( parse( text =   dim_tmp[2] ) )
        name_tmp <- trimws( which = "both", dim_tmp[1]   )
        
               dim_list  [[l]] <- val_tmp
        names( dim_list )[[l]] <- name_tmp
        
        assign( name_tmp, val_tmp )
      }
    }
  }

  #--- Compute indices ranges for ptracer array
  
  infile <- "../../../gud_1d_35+16/code_noradtrans/GUD_INDICES.h"
  if( !file.exists( infile ) ) {
    infile <- "../../../pkg/gud/GUD_INDICES.h"
  }
  df <- file( infile, "r" )
  
  ind_txt <- c( readLines( df, skipNul = TRUE ) )
  
  close( df )
  
  # Get indices values
  ind_1   <- trimws( ind_txt ) %>% substr( start = 1, stop = 1 )
  skip    <- ind_1 == "C" | # get rid of comments
             ind_1 == "c" |
             ind_1 == "&" |
             ind_1 == ""
  skip    <- skip                          | # avoid useless key word 
             grepl( "ALLOW_GUD", ind_txt )
  
  ind_txt <- ind_txt[!skip]
  
  # Take care of CPP optional key words
  ifdef   <- grep( "#ifdef|#ifndef", ind_txt)
  eldef   <- grep( "#else",          ind_txt)
  endef   <- grep( "#endif",         ind_txt)
  
  if( length(ifdef) != length(endef) ) { stop("Problem with CPP optional keys in *.h header files") }
  
  keep <- rep( T, length(ind_txt) )
  for(i in 1:length(ifdef) ) {
    
    id   <- sapply( names(opt), FUN = grepl, x = ind_txt[ ifdef[i] ] )
    
    test <- eval( parse( text = names(id)[id] ) )
    if( grepl( "ifndef", ind_txt[ ifdef[i] ] ) ) { test <- !test }
    
    keep[ ifdef[i] : endef[i] ] <- test
    
    id <- eldef > ifdef[i] & eldef < endef[i]
    if( any( id ) ) {
      keep[ eldef[id] : endef[i] ] <- !test
    }
  }
  keep[  grepl( "#", ind_txt ) ]                             <- FALSE
  keep[ !grepl( "PARAMETER", ind_txt, ignore.case = TRUE ) ] <- FALSE
  
  ind_txt <- ind_txt[ keep ]                                                  %>% # Keep parameter declarations only
              sub( pattern = "parameter", replacement = "", ignore.case = T ) %>%
             gsub( pattern = "&|[(]|[)]", replacement = ""   )                %>%
             gsub( pattern = "in",        replacement = "iN" )                %>% # issue with a FORTRAN variable name being an R "reserved" system word = "in"
             trimws( which = "both" )

  # Create list of dimensions and ranges
  ind_list <- list()
  
  l <- 0
  for( i in 1:length(ind_txt) ) {
    l <- l + 1
    
    # Check whether there is only one parameter per line...
    jx <- regexpr( ",", ind_txt[i] )
    
    # if so...
    if( jx == -1 ) {
      
      ind_tmp <- strsplit( ind_txt[i], "=" )
      
      # Add the parameter's name & value in the list returned
      val_tmp  <-   eval( parse( text =   ind_tmp[[1]][2] ) )
      name_tmp <- trimws( which = "both", ind_tmp[[1]][1]   )
      
             ind_list  [[l]] <- val_tmp
      names( ind_list )[[l]] <- name_tmp
      
      # Create the variable in the working space
      assign( name_tmp, val_tmp )
      
      # if not...
    } else {
      
      # ...split the parameters on the line
      ind_spl <- strsplit( ind_txt[i], "," ) %>% unlist()
      
      for( j in 1:length(ind_spl) ) {
        l <- l + j - 1
        
        ind_tmp  <- strsplit( ind_spl[j], "=" ) %>% unlist()
        
        val_tmp  <-   eval( parse( text =   ind_tmp[2] ) )
        name_tmp <- trimws( which = "both", ind_tmp[1]   )
        
               ind_list  [[l]] <- val_tmp
        names( ind_list )[[l]] <- name_tmp
        
        assign( name_tmp, val_tmp )
      }
    }
  }

  #--- Read parameters types & dimensions (definition)

  infile <- "../../../gud_1d_35+16/code_noradtrans/GUD_GENPARAMS.h"
  if( !file.exists( infile ) ) {
    infile <- "../../../pkg/gud/GUD_GENPARAMS.h"
  }
  df1 <- file( infile, "r" )
  
  infile <- "../../../gud_1d_35+16/code_noradtrans/GUD_TRAITS.h"
  if( !file.exists( infile ) ) {
    infile <- "../../../pkg/gud/GUD_TRAITS.h"
  }
  df2 <- file( infile, "r" )
  
  infile <- "../../../gud_1d_35+16/code_noradtrans/GUD_TRAITPARAMS.h"
  if( !file.exists( infile ) ) {
    infile <- "../../../pkg/gud/GUD_TRAITPARAMS.h"
  }
  df3 <- file( infile, "r" )
  
  def_txt <- c( readLines( df1, skipNul = TRUE ),
                readLines( df2, skipNul = TRUE ),
                readLines( df3, skipNul = TRUE ) )

  close( df1 ); close( df2 ); close( df3 )
  
  # Get parameters definitions
  def_1   <- trimws( def_txt ) %>% substr( start = 1, stop = 1 )
  skip    <- def_1 == "C" | # get rid of comments
             def_1 == "c" |
             def_1 == "&" |
             def_1 == ""
  skip    <- skip                          | # avoid useless key word 
             grepl( "ALLOW_GUD", def_txt )

  def_txt <- def_txt[!skip]
  
  # Take care of CPP optional key words
  ifdef   <- grep( "#ifdef|#ifndef", def_txt)
  eldef   <- grep( "#else",          def_txt)
  endef   <- grep( "#endif",         def_txt)
  
  if( length(ifdef) != length(endef) ) { stop("Problem with CPP optional keys in *.h header files") }
  
  keep <- rep( T, length(def_txt) )
  for(i in 1:length(ifdef) ) {

    id   <- sapply( names(opt), FUN = grepl, x = def_txt[ ifdef[i] ] )

    test <- eval( parse( text = names(id)[id] ) )
    if( grepl( "ifndef", def_txt[ ifdef[i] ] ) ) { test <- !test }
    
    keep[ ifdef[i] : endef[i] ] <- test

    id <- eldef > ifdef[i] & eldef < endef[i]
    if( any( id ) ) {
      keep[ eldef[id] : endef[i] ] <- !test
    }
  }
  keep[ grep( "#", def_txt ) ] <- FALSE
  
  def_txt <- def_txt[keep] %>% trimws( which = "both" )

  # Get *numerical* parameters definitions
  num_id  <- grepl( "INTEGER|^_RL", def_txt )

  num_txt <- def_txt[num_id]                                    %>% # Keep numerical values only
             sub( pattern = "INTEGER |_RL ", replacement = "" )     # remove declarations

  # Create parameters list
  def_list <- list()

  for( i in 1:length(num_txt) ) {
    
    # Get dimension(s) declaration: var(dim,...)
    num_tmp <- strsplit( num_txt[i], split = "[(]|[)]" ) %>% unlist() %>% trimws( which = "both" )

    # Check if parameter is scalar...
    if( length(num_tmp) == 1 ) {
             def_list  [[i]] <- 0
      names( def_list )[[i]] <- num_tmp[1]
    # ... or...
    } else {
      dim <- strsplit( num_tmp[2], "," ) %>% unlist() %>% trimws( which = "both" )
    # ... vector...
      if( length(dim) == 1 ) {
        id <- match( tolower(       dim       ),
                     tolower( names(dim_list) ) )
        num_dim <- dim_list[[id]]
    # ... array
      } else {
        num_dim <- NULL
           ndim <- length(dim)
        for( j in 1:ndim ) {
          id <- match( tolower(       dim[j]    ),
                       tolower( names(dim_list) ) )
          num_dim <- c( num_dim, dim_list[[id]] )
        }
      }
             def_list  [[i]] <- array( 0, dim = num_dim )
      names( def_list )[[i]] <- num_tmp[1]
    }
  }

  # Get *logical* variables definitions
  log_id  <- grepl( "LOGICAL ", def_txt )
  
  log_txt <- def_txt[log_id]                               %>% # Keep logical values only
             sub( pattern = "LOGICAL ", replacement = "" )     # remove declarations

  # Append list with logical values
  i0 <- length(def_list)
  for( i in 1:length(log_txt) ) {
             def_list  [[i0+i]] <- FALSE
      names( def_list )[[i0+i]] <- log_txt[i]
  }

  
  #--- Read default values for parameters
  
  infile <- "../../../gud_1d_35+16/code_noradtrans/gud_readparms.F"
  if( !file.exists( infile ) ) {
    infile <- "../../../pkg/gud/gud_readparms.F"
  }
  df1 <- file( infile, "r" )
  
  infile <- "../../../gud_1d_35+16/code_noradtrans/gud_readgenparams.F"
  if( !file.exists( infile ) ) {
    infile <- "../../../pkg/gud/gud_readgenparams.F"
  }
  df2 <- file( infile, "r" )
  
  infile <- "../../../gud_1d_35+16/code_noradtrans/gud_readtraitparams.F"
  if( !file.exists( infile ) ) {
    infile <- "../../../pkg/gud/gud_readtraitparams.F"
  }
  df3 <- file( infile, "r" )
  
  par_txt <- c( readLines( df1, skipNul = TRUE ),
                readLines( df2, skipNul = TRUE ),
                readLines( df3, skipNul = TRUE ) ) %>%
             trimws( which = "both" )
  
  close( df1 ); close( df2 ); close( df3 )

  # Ignore comment/empty lines
  par_1 <- substr( par_txt, 1, 1 )
  skip  <- par_1 == "&" | 
           par_1 == "C" | 
           par_1 == "c" | 
           par_1 == ""
  skip  <- skip                           |
           grepl( "ALLOW_GUD",  par_txt ) |
           grepl( "READ|WRITE", par_txt)
  
  par_txt  	<- par_txt[!skip]                               %>% # Ignore comment lines
               strsplit( split = "!" ) %>% sapply( "[", 1 ) %>% # Remove trailing comments
               trimws( which = "both" )                         # Trim white spaces
  
  # Take care of CPP optional key words
  ifdef   <- grep( "#ifdef|#ifndef", par_txt)
  eldef   <- grep( "#else",          par_txt)
  endef   <- grep( "#endif",         par_txt)
  
  if( length(ifdef) != length(endef) ) { stop("Problem with CPP optional keys in gud_read*.F  files") }
  
  keep <- rep( T, length(par_txt) )
  for(i in 1:length(ifdef) ) {
    
    i1   <- sapply( names(opt), FUN = grepl, x = par_txt[ ifdef[i] ] )

    test <- FALSE
    if( any(i1) ) {
        test <- eval( parse( text = names(i1)[i1] ) )
      if( grepl( "ifndef", par_txt[ ifdef[i] ] ) ) { 
        test <- !test 
      }
    }
    
    keep[ ifdef[i] : endef[i] ] <- test
    
    i2 <- eldef > ifdef[i] & eldef < endef[i]
    if( any(i1) & any(i2) ) {
      keep[ eldef[i2] : endef[i] ] <- !test
    }
  }

  # Keep only lines where values are assigned to variables
  keep[ !grepl( " = ", par_txt ) ] <- FALSE
  
  par_txt <- par_txt[keep]                                   %>%
             trimws( which = "both" )                        %>%
             gsub( pattern = " _d 0",   replacement = ""   ) %>% # Convert FORTRAN formats
             gsub( pattern = " _d -",   replacement = "e-" ) %>%
             gsub( pattern = ".FALSE.", replacement = "F"  ) %>%
             gsub( pattern = ".TRUE.",  replacement = "T"  )
  
  # Create the list of default variables
  par_list <- list()

  UNSET_RL <- UNINIT_RL <- UNSET_I <- UNINIT_I <- -999999999

  l <- 0
  for( i in 1:length(par_txt) ) {

    par_tmp <- strsplit( par_txt[i], "=" )
    
    if( length( par_tmp[[1]]) == 1 ) {
      warning( paste( "!!! No value assigned to variable", par_txt[i], "!!!" ) )
      next
    } else {
      
      # Assign the variable to the working space
      par_name  <- trimws( par_tmp[[1]][1], which = "both" )

      par_val   <- trimws( par_tmp[[1]][2], which = "both" )
      if( is.finite( as.numeric(par_val) ) ) {
        par_val <- as.numeric(par_val)
      }else {
        par_val <- tryCatch( eval( parse( text = par_val ) ), 
                             error = function(e) { return( UNSET_RL ) } )
      }
      assign( par_name, par_val )
      
      # Check if the variable is only updated
      k <- grep( paste0("^",par_name,"$"), names(par_list), ignore.case = TRUE )
      if( length(k) > 0 ) {
        # Update
        par_list[[k]]          <- par_val
      } else {
        # Append to list
                           l   <- l + 1
               par_list  [[l]] <- par_val
        names( par_list )[[l]] <- par_name
      }
    }
  }

  
  #--- Merge lists of parameter values
  
  params <- def_list
  l      <- length(params)
  
  id <- match( tolower( names(par_list) ),
               tolower( names(def_list) ) )

  for( i in 1:length(id) ) {
    
    if( is.finite(id[i]) ) {
      
      if( is.matrix( params[[id[i]]] ) ) {
        params[[id[i]]][,] <- unlist( par_list[i] )
      } else {
        params[[id[i]]][]  <- unlist( par_list[i] )
      }
      
    } else {
                       l   <- l + 1
             params  [[l]] <- par_list[[i]]
      names( params )[[l]] <- names(par_list[i])
    }
  }
  
  # Return merdged list with default parameter values
  params <- c( dim_list, ind_list, params )
  return(params)
}
