UNAME:=$(shell uname)
PWD:=$(shell pwd)

# intero and stack ghci are unable to use DYLD_LIBRARY_PATH on OSX
# See https://ghc.haskell.org/trac/ghc/ticket/11617
init:
	git submodule update --init --recursive
	( cd vendor; ./build-aten.sh )
#	( cd vendor; ./build-aten-spec.sh )
#	( cd vendor; ./build-error-handler.sh )
# ifeq ($(UNAME),Darwin)
#	sudo ln -sf $(PWD)/vendor/build/libATen.dylib /usr/local/lib/libATen.dylib
#	sudo ln -sf $(PWD)/vendor/build/libEHX.dylib /usr/local/lib/libEHX.dylib
#	@echo "\nCreated shared library symlinks for OSX:\n"
#	@sudo ls -l /usr/local/lib/libATen.dylib /usr/local/lib/libEHX.dylib
#	@echo
# endif
#	stack build

clean:
	rm -rf dist{,-newbuild)

purge:
	rm -rf vendor
	git checkout -- vendor
	git submodule update --init --recursive

build:
	cabal new-build all

refresh:
	cd output && ./refresh.sh

codegen-th:
	for l in TH THNN; do \
	  for t in generic concrete; do \
	    cabal new-run hasktorch-codegen:ht-codegen -- --type $$t --lib $$l --verbose; \
	  done; \
	done

codegen-thc:
	for l in THC THCUNN; do for t in generic concrete; do cabal new-run hasktorch-codegen:ht-codegen -- --type $$t --lib $$l --verbose; done; done

codegen: codegen-th codegen-thc
codegen-refresh: codegen refresh

dev:
	sos -e 'dist' -p '.*hsig$$' -p '.*hs$$' -p '.*cabal$$' -p 'cabal.project$$' -c 'cabal new-build all'

# lasso is broken
run-examples:
	for ex in ad \
	  bayesian-regression \
	  download-mnist \
	  ff-typed \
	  ff-untyped \
	  gradient-descent \
	  multivariate-normal \
	  static-tensor-usage; do \
	    echo "running $$ex" && \
	    sleep 1 && \
	    cabal new-run hasktorch-examples:$$ex && \
	    sleep 1 ; \
	done
	echo "finished running examples"

dabal-all:
	for lib in "signatures/hasktorch-signatures"        \
	  "signatures/partial/hasktorch-signatures-partial" \
	  "signatures/support/hasktorch-signatures-support" \
	  "signatures/types/hasktorch-signatures-types"     \
	  "indef/hasktorch-indef"                           \
	  "hasktorch/hasktorch"                             \
	  "zoo/hasktorch-zoo"                               \
	  "examples/hasktorch-examples"  ; do               \
	  $(MAKE) dabal DABAL=$${lib} ; \
	done

dabal: dabal-tmp dabal-tmp-switch

dabal-tmp:
	dhall-to-cabal $(DABAL).dhall > $(DABAL)-tmp.cabal

dabal-tmp-switch:
	mv $(DABAL)-tmp.cabal $(DABAL).cabal

test-signatures:
	for pkg in \
	  floating-th \
	  floating-thc \
	  signed-th \
	  signed-thc \
	  unsigned-thc \
	  unsigned-th; do \
	    cabal new-build hasktorch-signatures:isdefinite-$${pkg} > /dev/null ; \
	done

.PHONY: clean build refresh codegen init dev run-examples
