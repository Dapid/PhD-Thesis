all:
	latexmk -lualatex -latexoption="-synctex=1" main.tex 

clean:
	latexmk -c

purge:
	latexmk -C

count:
	@echo -n "TODO: "
	@find . -name "*.tex" | xargs grep -F "\todo" | wc -l
	@echo -n "FIGS: "
	@find . -name "*.tex" | xargs grep -F "\missingfigure" | wc -l

