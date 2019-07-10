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

check:
	@find . -name "*tex" | xargs -I [] grep \" [] | grep -v "\`\`" && exit 1 || exit 0

txt:
	pandoc -f latex -t plain -o thesis.txt main.tex

