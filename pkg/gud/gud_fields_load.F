#include "GUD_OPTIONS.h"
#include "EXF_OPTIONS.h"

CBOP
C !ROUTINE: GUD_FIELDS_LOAD

C !INTERFACE: ==========================================================
      subroutine gud_fields_load( mytime, myiter, mythid )

C !DESCRIPTION:
C     Load and interpolate forcing fields for gud package

C !USES: ===============================================================
      IMPLICIT NONE
#include "SIZE.h"
#include "EEPARAMS.h"
#include "PARAMS.h"
#include "GRID.h"
#ifdef ALLOW_EXF
#include "EXF_PARAM.h"
#endif
#ifdef ALLOW_GUD
#include "GUD_SIZE.h"
#include "GUD_EXF.h"
#include "GUD_TRAITS.h"
#endif

C !INPUT PARAMETERS: ===================================================
      _RL     mytime
      integer myiter
      integer mythid
CEOP

#ifdef ALLOW_GUD
#ifdef ALLOW_EXF

C !LOCAL VARIABLES: ====================================================
      INTEGER bi,bj

c--   read forcing fields from files and temporal interpolation

      IF (.NOT.(gud_loadFieldsEarly.AND.myiter.EQ.niter0)) THEN
        CALL GUD_EXF_LOAD(myTime, myIter, myThid)
      ENDIF

      CALL GUD_MONITOR(myTime, myIter, myThid)

#endif /* ALLOW_EXF */

#ifdef ALLOW_DIAGNOSTICS
      IF (useDIAGNOSTICS) THEN
       CALL DIAGNOSTICS_FILL(surfPAR,'exfPAR  ',1,1,0,0,0,myThid)
      ENDIF
#endif

#endif /* ALLOW_GUD */

      RETURN
      END
