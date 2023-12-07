<?php
$identifier = $_POST['identifier'];
$bpm = $_POST['bpm'];
$path = ".\\temp\\".$identifier."\\";
mkdir($path, 0700, true);

$path = $path.$_FILES['file']['name'];

move_uploaded_file($_FILES['file']['tmp_name'], $path);

$command = '.\venv\Scripts\python.exe extract.py "'.$identifier.'" "'.$path.'" "'.$bpm.'"';

echo $identifier."\n";
echo $bpm."\n";
echo $command."\n";	

shell_exec($command);
?>