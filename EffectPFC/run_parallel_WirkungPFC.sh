startTime=$(date +%s)

python create_stim.py 1
python create_stim.py 2
python create_stim.py 3

let parallel=8
let durchgaenge=15

for durchgang in $(seq $durchgaenge); do
	startdurchgangTime=$(date +%s)
        for i in $(seq $parallel); do
                let y=$i+$parallel*$((durchgang - 1))
                python run_cla.py $y 0 &
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


wait
for durchgang in $(seq $durchgaenge); do
	startdurchgangTime=$(date +%s)
        for i in $(seq $parallel); do
                let y=$i+$parallel*$((durchgang - 1))
                python run_cla.py $y 1 &
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


wait
for durchgang in $(seq $durchgaenge); do
	startdurchgangTime=$(date +%s)
        for i in $(seq $parallel); do
                let y=$i+$parallel*$((durchgang - 1))
                python run_cla.py $y 2 &
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

