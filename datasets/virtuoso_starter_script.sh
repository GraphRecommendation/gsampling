#!/bin/bash
cd /opt/virtuoso-opensource
ARG=( "ld_dir('/import', '*.nt', 'http://localhost:8890/dataspace');" "rdf_loader_run();" "checkpoint;" "exit;" )
for a in "${ARG[@]}"; do
  echo "$a" | /opt/virtuoso-opensource/bin/isql 1111 ;
done
exit()