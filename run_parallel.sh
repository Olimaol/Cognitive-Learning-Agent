startTime=$(date +%s)

let parallel=10
let durchgaenge=2

for durchgang in $(seq $durchgaenge); do
	startdurchgangTime=$(date +%s)
        for i in $(seq $parallel); do
                let y=$i+$parallel*$((durchgang - 1))
                python run_cla_Training.py $y 2 &
        done
        wait
        sleep 5
	echo $durchgang >> fertig.txt
	endTime=$(date +%s)
	dif=$((endTime - startdurchgangTime))
	echo $dif >> zeit.txt
	sleep 5
done

endTime=$(date +%s)
dif=$((endTime - startTime))
echo $dif >> zeit.txt

