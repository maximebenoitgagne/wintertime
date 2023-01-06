
def julian_day(i, j, k):
    '''
    This function converts a calendar date to the corresponding Julian
    day starting at noon on the calendar date.  The algorithm used is
    from Van Flandern and Pulkkinen, Ap. J. Supplement Series 41,
    November 1979, p. 400.

    Written by Frederick S. Patt, GSC, November 4, 1992

    i :: Year - e.g. 1970
    j :: Month - (1-12)
    k :: Day  - (1-31)

    also works for i=year, j=1, k=yearday
    '''

    jd = 367*i - 7*(i+(j+9)//12)//4 + 275*j//9 + k + 1721014

    # This additional calculation is needed only for dates outside of the
    # period March 1, 1900 to February 28, 2100
#    julian_day = julian_day + 15 - 3*((i+(j-9)//7)//100+1)//4

    return jd

