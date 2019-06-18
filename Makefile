all:
	latexmk -lualatex -latexoption="-synctex=1" main.tex 

clean:
	latexmk -c

purge:
	latexmk -C

