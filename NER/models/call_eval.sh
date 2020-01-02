cat $1 | awk '{print $1" "$2" "$3}' | ../conlleval
