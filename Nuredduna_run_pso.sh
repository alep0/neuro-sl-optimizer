#!/usr/bin/sh

#chmod +x scripts/Nuredduna_run_pso.sh
#chmod +x scripts/run_pso.sh
#./scripts/Nuredduna_run_pso.sh

# "R01"
# "1 2"
# "1"
# "3"
# "1"

####################################
# Call Python script via bash script

echo $1 $2 $3 $4 $5

rats=$1
rea=$2
oc=$3
on=$4
om=$5

# Parámetros ajustables
SLEEP_BETWEEN_POLLS=5
MAX_ATTEMPTS=0   # 0 significa reintento indefinido

for r in $rea;
  do
  for rat in $rats;
    do

    attempt=0

    while :; do
      attempt=$((attempt + 1))
      echo "Enviando job para t=$t (intento $attempt)"

      MY_JOB="./scripts/run_pso.sh --realizations \"$r\" --op-corr \"$oc\" --op-net \"$on\" --op-model \"$om\""

      # lanzar el job y capturar la línea "Submitted batch job <id>"
      SUB_OUT=$(run -t 123:30 -c 1 -m 16 -j run_pso_c1_m16_"$rat" "$MY_JOB" 2>&1)

      RETCODE=$?
      echo $SUB_OUT

      if [ $RETCODE -ne 0 ]; then
        echo "El comando run devolvió código $RETCODE. Salida:"
        echo "$SUB_OUT"
        if [ "$MAX_ATTEMPTS" -ne 0 ] && [ "$attempt" -ge "$MAX_ATTEMPTS" ]; then
          echo "Máximo de intentos alcanzado para t=$t. Abortando este t."
          break
        fi
        sleep $SLEEP_BETWEEN_POLLS
        continue
      fi

      JOBID=$(printf "%s\n" "$SUB_OUT" | awk '/Submitted batch job/ {print $NF; exit}')
      if [ -z "$JOBID" ]; then
        echo "No se obtuvo jobid al enviar el job. Salida de run:"
        echo "$SUB_OUT"
        if [ "$MAX_ATTEMPTS" -ne 0 ] && [ "$attempt" -ge "$MAX_ATTEMPTS" ]; then
          echo "Máximo de intentos alcanzado para t=$t. Abortando este t."
          break
        fi
        sleep $SLEEP_BETWEEN_POLLS
        continue
      fi

      LOGFILE_e="$(pwd)/scripts/run_pso.sh.e${JOBID}"
      LOGFILE_o="$(pwd)/scripts/run_pso.sh.o${JOBID}"
      echo "Job enviado: $JOBID. Esperando a que aparezca el log $LOGFILE_o..."

      # Esperar a que el log exista y que el job termine de escribir (polling)
      while [ ! -f "$LOGFILE_o" ]; do
        echo $SLEEP_BETWEEN_POLLS
        sleep $SLEEP_BETWEEN_POLLS
      done

      # Poll del contenido hasta detectar uno de los dos patrones
      while :; do
        echo "Buscando patrones ..."
        if grep -q -E "Illegal instruction" "$LOGFILE_e" || grep -q -E "Exited with exit code 132" "$LOGFILE_e"; then
          echo "Detectado 'Illegal instruction' en $LOGFILE_e para t=$t. Reintentando en otro nodo..."
          break
        fi

        if grep -q "| INFO     | source.core.simulation_engine | Backend: C++ (accelerated) " "$LOGFILE_e"; then
          echo "Job t=$t finalizó correctamente según $LOGFILE_e. Continuando con el siguiente t."
          break 2
        fi

        if grep -q -E "WARNING:root:C++ module not available, using pure Python" "$LOGFILE_o"; then
          echo "Detectado 'WARNING:root:C++ module not available, using pure Python' en $LOGFILE_o para t=$t. Reintentando en otro nodo..."
          break
        fi

        # Comprobar si el job terminó sin ninguno de los patrones y sin éxito claro
        # Si el job ya no está en la cola y no apareció ninguno de los patrones, tratar como fallo y reintentar
        if ! squeue -j "$JOBID" -h >/dev/null 2>&1; then
          # si no hay señales de success ni de illegal instruction, mostrar últimas líneas y reintentar
          echo "Job $JOBID ya no está en cola y no se detectó éxito. Últimas líneas de $LOGFILE:"
          tail -n 40 "$LOGFILE"
          break
        fi

        sleep $SLEEP_BETWEEN_POLLS
      done

      # Comprobar límite de reintentos
      if [ "$MAX_ATTEMPTS" -ne 0 ] && [ "$attempt" -ge "$MAX_ATTEMPTS" ]; then
        echo "Máximo de intentos alcanzado ($MAX_ATTEMPTS) para t=$t. Abortando este t."
        break
      fi

      # Si se llegó aquí por "Illegal instruction" el while exterior continuará e intentará de nuevo
      # Si se salió con éxito el break 2 ya hizo continuar con el siguiente t
      sleep $SLEEP_BETWEEN_POLLS
    done
  done
done
