import React from 'react'
import Grid from '@material-ui/core/Grid'
import Input from '@material-ui/core/Input'
import Slider from '@material-ui/core/Slider'
import ListItem from '@material-ui/core/ListItem'
import Typography from '@material-ui/core/Typography'
import { makeStyles } from '@material-ui/core/styles'

const useStyles = makeStyles({
  root: {
    width: 240
  },
  input: {
    width: 50
  }
})

type FolderSliderProps = {
    label: string
    value: number | string | Array<number | string>
    setValue(value: number | number[]): void
    step: number
    min: number
    max: number
}

export default function FolderSlider (props: FolderSliderProps): React.ReactNode {
  const classes = useStyles()

  const handleSliderChange = (event: any, newValue: number | number[]): void => {
    props.setValue(newValue)
  }

  const handleInputChange = (event: React.ChangeEvent<HTMLInputElement>): void => {
    if (event.target.value === '') {
    } else {
      props.setValue(Number(event.target.value))
    }
  }

  const handleBlur = () => {
    if (props.value < props.min) {
      props.setValue(props.min)
    } else if (props.value > props.max) {
      props.setValue(props.max)
    }
  }

  return (
    <div className={classes.root}>
      <ListItem button key={props.label}>
        <Grid container spacing={2} alignItems='center'>
          <Grid item>
            <Typography variant='body2' noWrap>{props.label}</Typography>
          </Grid>
          <Grid item xs>
            <Slider
              value={typeof props.value === 'number' ? props.value : 0}
              min={props.min}
              max={props.max}
              step={props.step}
              onChange={handleSliderChange}
              aria-labelledby='input-slider'
            />
          </Grid>
          <Grid item>
            <Input
              className={classes.input}
              value={props.value}
              onChange={handleInputChange}
              onBlur={handleBlur}
              inputProps={{
                step: props.step,
                min: props.min,
                max: props.max,
                type: 'number',
                'aria-labelledby': 'input-slider'
              }}
            />
          </Grid>
        </Grid>
      </ListItem>
    </div>
  )
}

FolderSlider.defaultProps = {
  label: 'Volume',
  step: 10,
  min: 0,
  max: 100,
  value: 30,
  setValue: (value: number | number[]) => { }
}
