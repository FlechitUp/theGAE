#!/bin/bash

# Definir la ruta al script de Python
PYTHON_SCRIPT="theGAE.py -csv -dd"

# Lista de valores
HIDD_DIM=(200 150 100)
OUT_DIM=(50 25 10)
EPOCHS=(100 500 1000)

# Número máximo de ejecuciones
MAX_RUNS=3

# Bucle que se ejecuta hasta alcanzar el número máximo de ejecuciones
for ((i=0; i<MAX_RUNS; i++))
do
    HD_CURRENT=${HIDD_DIM[$i]}

    for ((j=0; j<MAX_RUNS; j++))
    do
        OD_CURRENT=${OUT_DIM[$j]}
        for ((k=0; k<MAX_RUNS; k++))
        do
            E_CURRENT=${EPOCHS[$k]}
            echo "Execución $((i)):"
    
            # Ejecutar el script de Python
            echo "python3 $PYTHON_SCRIPT $HD_CURRENT $OD_CURRENT $E_CURRENT"
            python3 $PYTHON_SCRIPT $HD_CURRENT $OD_CURRENT $E_CURRENT &

            # Esperar a que la ejecución actual termine
            wait
    
            # Comprobar si es la última ejecución
            if [ $i -lt $((MAX_RUNS-1)) ]
            then
                echo "Esperando antes de la siguiente ejecución..."
                sleep 15  # Esperar n segundos 
            fi
        done
    done
done

echo "Script completado."
